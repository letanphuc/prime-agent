//! Claude Agent provider: runs the user's own `claude` (Claude Code) as the model backend.
//!
//! Prime Agent never sees a Claude token: `claude` signs in, refreshes its login and calls
//! Anthropic itself (`claude auth login` in a terminal). Prime Agent drives it over the
//! stream-json protocol, offers its own tools as an in-process MCP server, and turns Claude's
//! streamed events into Prime Agent stream events. Claude Code's built-in tools, settings files,
//! CLAUDE.md files, hooks, skills, auto memory and auto compaction are turned off.
//!
//! The existing `anthropic` provider is untouched; this is a separate provider whose models
//! mirror the Anthropic catalog under the `claude-agent` provider id.

use std::path::{Path, PathBuf};

use serde_json::Map;

use crate::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEvent, AssistantMessageEventStream,
};
use crate::registry::Provider;
use crate::types::{
    done_reason, error_reason, AssistantMessage, Context, Model, ModelThinkingLevel,
    SimpleStreamOptions, StopReason, StreamOptions, Usage,
};
use crate::utils_inner::diagnostics::now_ms;
use crate::utils_inner::stream_failure::format_stream_failure_message;
use crate::utils_inner::ProviderError;

mod content;
mod convert;
mod fingerprint;
mod process;
mod sessions;

#[cfg(test)]
mod tests;

/// The API id of the Claude Agent provider.
pub const API_CLAUDE_AGENT: &str = "claude-agent";
/// The provider id of the Claude Agent models.
pub const CLAUDE_AGENT_PROVIDER_ID: &str = "claude-agent";
/// Overrides which `claude` executable runs (default: `claude` on `PATH`).
pub const CLAUDE_PATH_ENV: &str = "PRIME_AGENT_CLAUDE_PATH";
/// The base URL the derived models carry: Claude Code makes the API calls, not Prime Agent.
const LOCAL_BASE_URL: &str = "claude-code://local";

fn executable_in(dir: &Path) -> Option<PathBuf> {
    let candidate = dir.join(if cfg!(windows) {
        "claude.exe"
    } else {
        "claude"
    });
    candidate.is_file().then_some(candidate)
}

/// The `claude` executable to run: `PRIME_AGENT_CLAUDE_PATH`, then `claude` on `PATH`, then the
/// installer's default locations.
#[must_use]
pub fn find_claude_executable() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(CLAUDE_PATH_ENV).filter(|value| !value.is_empty()) {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    if let Some(found) = std::env::var_os("PATH")
        .and_then(|path| std::env::split_paths(&path).find_map(|dir| executable_in(&dir)))
    {
        return Some(found);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    [home.join(".local/bin"), home.join(".claude/local")]
        .iter()
        .find_map(|dir| executable_in(dir))
}

/// The Claude Agent copy of an Anthropic catalog model: same id, limits and prices, served
/// through `claude`.
#[must_use]
pub fn claude_agent_model(anthropic: &Model) -> Model {
    let mut model = anthropic.clone();
    model.provider = CLAUDE_AGENT_PROVIDER_ID.to_string();
    model.api = API_CLAUDE_AGENT.to_string();
    model.base_url = LOCAL_BASE_URL.to_string();
    model.headers = None;
    model.compat = None;
    model
}

fn effort(level: Option<ModelThinkingLevel>) -> Option<&'static str> {
    match level? {
        ModelThinkingLevel::Off => None,
        ModelThinkingLevel::Minimal | ModelThinkingLevel::Low => Some("low"),
        ModelThinkingLevel::Medium => Some("medium"),
        ModelThinkingLevel::High => Some("high"),
        ModelThinkingLevel::Xhigh => Some("xhigh"),
        ModelThinkingLevel::Max => Some("max"),
    }
}

fn error_message(model: &Model, text: String) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        stop_reason_raw: None,
        error_message: Some(text),
        timestamp: now_ms(),
        rest: Map::default(),
    }
}

/// Stream one completion through `claude`.
#[must_use]
pub fn stream_claude_agent(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let (writer, reader) = create_assistant_message_event_stream();
    let Some(executable) = find_claude_executable() else {
        let message = error_message(
            model,
            "Claude Code is not installed. Install it (https://code.claude.com/docs/en/setup), then run `claude auth login`."
                .to_string(),
        );
        writer.push(AssistantMessageEvent::Error {
            reason: crate::types::ErrorStopReason::Error,
            error: message.clone(),
        });
        writer.end(Some(message));
        return reader;
    };
    let base = options
        .map(|options| options.base.clone())
        .unwrap_or_default();
    let request = sessions::Request {
        executable,
        session_id: base.session_id.clone(),
        model: model.clone(),
        effort: effort(options.and_then(|options| options.reasoning)),
        system_prompt: context.system_prompt.clone(),
        messages: context.messages.clone(),
        tools: context.tools.clone().unwrap_or_default(),
        signal: base.signal,
    };
    tokio::spawn(async move {
        let mut converter = convert::Converter::new(&request.model, &request.tools, false);
        writer.push(AssistantMessageEvent::Start {
            partial: converter.output.clone(),
        });
        let outcome = sessions::run(&request, &mut converter, &writer).await;
        let mut output = converter.output;
        match outcome {
            Ok(_) => {
                writer.push(AssistantMessageEvent::Done {
                    reason: done_reason(output.stop_reason),
                    message: output,
                });
                writer.end(None);
            }
            Err(error) => {
                output.stop_reason = if error == ProviderError::Aborted {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                output.error_message = Some(format_stream_failure_message(&error));
                writer.push(AssistantMessageEvent::Error {
                    reason: error_reason(output.stop_reason),
                    error: output.clone(),
                });
                writer.end(Some(output));
            }
        }
    });
    reader
}

/// Registry provider for the `claude-agent` API.
pub struct ClaudeAgentProvider;

impl Provider for ClaudeAgentProvider {
    fn api(&self) -> &str {
        API_CLAUDE_AGENT
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let options = options.map(|base| SimpleStreamOptions::from_base(base.clone()));
        stream_claude_agent(model, context, options.as_ref())
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        stream_claude_agent(model, context, options)
    }
}
