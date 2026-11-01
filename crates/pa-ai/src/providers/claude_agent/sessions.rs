//! Long-running `claude` processes, one per Prime Agent session.
//!
//! A tool boundary ends only the Prime Agent stream: the Claude turn stays open, its `tools/call`
//! requests parked, until the next stream call brings the results. Prime Agent's history is the
//! source of truth: whatever `claude` has not seen reaches it inside a conversation checkpoint,
//! and a changed history starts a new Claude session. A process stops after five idle minutes;
//! the next turn resumes its saved session.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use super::content::{catch_up, collect_results, content};
use super::convert::{Converter, Outcome};
use super::fingerprint::{checkpoints, decide, Decision, SessionRecord};
use super::process::{
    new_session_id, ClaudeProcess, Inbound, Launch, Persistence, SystemPrompt, ToolCallRequest,
};
use crate::event_stream::AssistantMessageEventWriter;
use crate::types::{Message, Model, Tool};
use crate::utils_inner::ProviderError;

const IDLE: Duration = Duration::from_mins(5);
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
const RUNTIME_NOTE: &str = "<runtime_info>You are running in Prime Agent through the Claude Code harness. Prime Agent's tools are offered as MCP tools named mcp__prime__<tool>. Mention this only when asked.</runtime_info>";

/// One stream call.
pub(crate) struct Request {
    pub executable: std::path::PathBuf,
    /// Chat turns have one; side calls (no session) run in a throwaway process.
    pub session_id: Option<String>,
    pub model: Model,
    pub effort: Option<&'static str>,
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub signal: Option<CancellationToken>,
}

struct Live {
    process: ClaudeProcess,
    model: String,
    effort: Option<&'static str>,
    /// Settings that need a new process to change.
    launch: String,
    /// Offered-tool calls of the parked turn, waiting for Prime Agent's results.
    waiting: Option<Vec<String>>,
    /// Results that arrived before Claude Code asked for them.
    results: HashMap<String, Value>,
    /// Calls Claude Code made before their results arrived.
    parked: HashMap<String, ToolCallRequest>,
}

#[derive(Default)]
struct Entry {
    live: Option<Live>,
    record: Option<SessionRecord>,
    idle_generation: u64,
}

type Shared = Arc<tokio::sync::Mutex<Entry>>;

fn entries() -> &'static std::sync::Mutex<HashMap<String, Shared>> {
    static ENTRIES: OnceLock<std::sync::Mutex<HashMap<String, Shared>>> = OnceLock::new();
    ENTRIES.get_or_init(Default::default)
}

fn entry_for(session_id: Option<&str>) -> Shared {
    let Some(session_id) = session_id else {
        return Shared::default();
    };
    entries()
        .lock()
        .expect("claude-agent entries lock")
        .entry(session_id.to_string())
        .or_default()
        .clone()
}

fn launch_key(request: &Request) -> String {
    json!([request.system_prompt, request.tools]).to_string()
}

fn system_prompt(request: &Request) -> SystemPrompt {
    let prompt = request.system_prompt.clone().unwrap_or_default();
    if request.session_id.is_none() {
        return SystemPrompt::Replace(if prompt.is_empty() {
            "You are a text generation helper. Return only the requested output.".to_string()
        } else {
            prompt
        });
    }
    if prompt.is_empty() {
        SystemPrompt::Append(RUNTIME_NOTE.to_string())
    } else {
        SystemPrompt::Append(format!("{RUNTIME_NOTE}\n\n{prompt}"))
    }
}

async fn stop(entry: &mut Entry) {
    if let Some(mut live) = entry.live.take() {
        cancel_parked(&mut live).await;
        live.process.stop().await;
    }
}

/// Answer every parked call so Claude Code is not left waiting.
async fn cancel_parked(live: &mut Live) {
    for (_, call) in live.parked.drain() {
        let cancelled =
            json!({"content": [{"type": "text", "text": "Cancelled by user."}], "isError": true});
        // A failed write means the process is gone; nothing waits on the answer then.
        let _ = live.process.answer_tool(&call, &cancelled).await;
    }
    live.results.clear();
    live.waiting = None;
}

fn arm_idle(shared: &Shared, entry: &mut Entry) {
    entry.idle_generation += 1;
    let generation = entry.idle_generation;
    let shared = shared.clone();
    tokio::spawn(async move {
        tokio::time::sleep(IDLE).await;
        let mut entry = shared.lock().await;
        if entry.idle_generation == generation {
            stop(&mut entry).await;
        }
    });
}

async fn open(
    entry: &mut Entry,
    request: &Request,
    persistence: Persistence,
) -> Result<(), String> {
    if entry.live.is_none() {
        let process = ClaudeProcess::start(&Launch {
            executable: request.executable.clone(),
            model: request.model.id.clone(),
            effort: request.effort,
            system: system_prompt(request),
            tools: request.tools.clone(),
            persistence,
        })
        .await?;
        entry.live = Some(Live {
            process,
            model: request.model.id.clone(),
            effort: request.effort,
            launch: launch_key(request),
            waiting: None,
            results: HashMap::new(),
            parked: HashMap::new(),
        });
    }
    let live = entry.live.as_mut().expect("opened above");
    // Model and effort change on the running process.
    if live.model != request.model.id {
        live.process
            .request(json!({"subtype": "set_model", "model": request.model.id}))
            .await?;
        live.model.clone_from(&request.model.id);
    }
    if live.effort != request.effort {
        live.process
            .request(json!({"subtype": "apply_flag_settings", "settings": {"effortLevel": request.effort}}))
            .await?;
        live.effort = request.effort;
    }
    Ok(())
}

/// Where the turn starts and, for a parked turn, the tool results that continue it.
struct Prepared {
    start: usize,
    continuation: Option<(usize, HashMap<String, Value>)>,
}

async fn prepare(entry: &mut Entry, request: &Request) -> Prepared {
    // Retire an exited or incompatible process before continuing anything on it.
    let restart = entry
        .live
        .as_ref()
        .is_some_and(|live| live.process.ended() || live.launch != launch_key(request));
    if restart {
        stop(entry).await;
    }
    if request.session_id.is_none() {
        return Prepared {
            start: 0,
            continuation: None,
        };
    }
    let start = match decide(&request.messages, entry.record.as_ref()) {
        Decision::Fresh | Decision::Changed => {
            stop(entry).await;
            entry.record = None;
            0
        }
        Decision::Continue { start } | Decision::CatchUp { start } => start,
    };
    let Some(waiting) = entry.live.as_ref().and_then(|live| live.waiting.clone()) else {
        return Prepared {
            start,
            continuation: None,
        };
    };
    let (results, end) = collect_results(&request.messages, start);
    if waiting.iter().any(|id| !results.contains_key(id)) {
        // The parked turn cannot get its results: start over with the whole history.
        stop(entry).await;
        entry.record = None;
        return Prepared {
            start: 0,
            continuation: None,
        };
    }
    Prepared {
        start,
        continuation: Some((end, results)),
    }
}

/// Run one stream call. Returns the outcome; the converter holds the assistant message.
pub(crate) async fn run(
    request: &Request,
    converter: &mut Converter<'_>,
    writer: &AssistantMessageEventWriter,
) -> Result<Outcome, ProviderError> {
    let shared = entry_for(request.session_id.as_deref());
    let mut entry = shared.lock().await;
    entry.idle_generation += 1;
    let result = turn(&mut entry, request, converter, writer).await;
    if request.session_id.is_none() {
        stop(&mut entry).await;
    } else {
        let parked = entry
            .live
            .as_ref()
            .is_some_and(|live| live.waiting.is_some());
        // A turn waiting for tool results is busy however long the tools take.
        if !parked {
            arm_idle(&shared, &mut entry);
        }
    }
    result
}

// Long by design: one turn's prepare, open, input and record steps share the entry borrow.
#[allow(clippy::too_many_lines)]
async fn turn(
    entry: &mut Entry,
    request: &Request,
    converter: &mut Converter<'_>,
    writer: &AssistantMessageEventWriter,
) -> Result<Outcome, ProviderError> {
    let mut prepared = prepare(entry, request).await;
    let persistence = match (&request.session_id, &entry.record) {
        (None, _) => Persistence::Discard,
        (Some(_), Some(record)) if entry.live.is_none() => {
            Persistence::Resume(record.claude_session_id.clone())
        }
        (Some(_), _) => Persistence::New(new_session_id()),
    };
    let resuming = matches!(persistence, Persistence::Resume(_));
    if let Err(error) = open(entry, request, persistence).await {
        if !resuming {
            return Err(ProviderError::Message(error));
        }
        // The saved session is gone or unreadable: start over with the whole history.
        tracing::warn!(
            error,
            "claude-agent: resume failed; starting a new Claude session"
        );
        entry.live = None;
        entry.record = None;
        prepared = Prepared {
            start: 0,
            continuation: None,
        };
        open(entry, request, Persistence::New(new_session_id()))
            .await
            .map_err(ProviderError::Message)?;
    }
    let blocks = match (&request.session_id, &entry.record, &prepared.continuation) {
        (_, _, Some((end, _))) => catch_up(&request.messages, *end),
        (Some(_), _, None) | (None, Some(_), None) => catch_up(&request.messages, prepared.start),
        (None, None, None) => content(&request.messages),
    };
    let live = entry.live.as_mut().expect("opened above");
    converter.set_subscription(live.process.subscription());
    match prepared.continuation.take() {
        None => {
            let blocks = if blocks.is_empty() {
                vec![json!({"type": "text", "text": "Continue."})]
            } else {
                blocks
            };
            live.process
                .send_user(blocks, false)
                .await
                .map_err(ProviderError::Message)?;
        }
        Some((_, results)) => {
            if !blocks.is_empty() {
                live.process
                    .send_user(blocks, true)
                    .await
                    .map_err(ProviderError::Message)?;
            }
            live.waiting = None;
            live.results.extend(results);
            let ready: Vec<String> = live
                .parked
                .keys()
                .filter(|id| live.results.contains_key(*id))
                .cloned()
                .collect();
            for id in ready {
                let call = live.parked.remove(&id).expect("listed above");
                let result = live.results.remove(&id).expect("listed above");
                live.process
                    .answer_tool(&call, &result)
                    .await
                    .map_err(ProviderError::Message)?;
            }
        }
    }

    let outcome = read_step(live, request.signal.as_ref(), converter, writer).await;
    let claude_session_id = live.process.session_id.clone();
    match &outcome {
        Ok(Outcome::Tools) => live.waiting = Some(converter.tool_call_ids()),
        Ok(Outcome::Finished) => {
            live.waiting = None;
            live.results.clear();
        }
        Err(_) => {}
    }
    if outcome.is_err() && live.process.ended() {
        entry.live = None;
    }
    if let (Some(_), Some(claude_session_id)) = (&request.session_id, claude_session_id) {
        let reply = Message::Assistant(converter.output.clone());
        entry.record = Some(match &outcome {
            Ok(_) => SessionRecord {
                claude_session_id,
                history: checkpoints(&request.messages, Some(&reply)),
                interrupted: false,
            },
            Err(ProviderError::Aborted) => SessionRecord {
                claude_session_id,
                history: checkpoints(&request.messages, None),
                interrupted: true,
            },
            // A failed turn leaves no reliable history; the next turn rebuilds it.
            Err(_) => SessionRecord {
                claude_session_id,
                history: Vec::new(),
                interrupted: false,
            },
        });
    }
    outcome
}

async fn read_step(
    live: &mut Live,
    signal: Option<&CancellationToken>,
    converter: &mut Converter<'_>,
    writer: &AssistantMessageEventWriter,
) -> Result<Outcome, ProviderError> {
    let never = CancellationToken::new();
    let signal = signal.unwrap_or(&never);
    loop {
        let inbound = tokio::select! {
            biased;
            () = signal.cancelled() => {
                interrupt_and_drain(live).await;
                return Err(ProviderError::Aborted);
            }
            inbound = live.process.inbound.recv() => inbound,
        };
        match inbound {
            None => {
                return Err(ProviderError::Message(
                    "Claude Code stopped before finishing its reply".to_string(),
                ))
            }
            Some(Inbound::Exited(stderr)) => {
                let detail = if stderr.is_empty() {
                    String::new()
                } else {
                    format!(": {stderr}")
                };
                return Err(ProviderError::Message(format!(
                    "Claude Code stopped before finishing its reply{detail}"
                )));
            }
            Some(Inbound::ToolCall(call)) => {
                let result = call
                    .tool_use
                    .as_ref()
                    .and_then(|id| live.results.remove(id));
                match (result, call.tool_use.clone()) {
                    (Some(result), _) => live
                        .process
                        .answer_tool(&call, &result)
                        .await
                        .map_err(ProviderError::Message)?,
                    (None, Some(id)) => {
                        live.parked.insert(id, call);
                    }
                    (None, None) => {
                        let error = json!({"content": [{"type": "text", "text": "Prime Agent could not match this tool call to its result."}], "isError": true});
                        live.process
                            .answer_tool(&call, &error)
                            .await
                            .map_err(ProviderError::Message)?;
                    }
                }
            }
            Some(Inbound::Message(message)) => {
                if let Some(outcome) = converter
                    .handle(&message, writer)
                    .map_err(ProviderError::Message)?
                {
                    return Ok(outcome);
                }
            }
        }
    }
}

/// Stop the running Claude turn and wait for its result, so the next turn starts clean.
async fn interrupt_and_drain(live: &mut Live) {
    if live.process.interrupt().await.is_err() {
        return;
    }
    cancel_parked(live).await;
    let drain = async {
        while let Some(inbound) = live.process.inbound.recv().await {
            match inbound {
                Inbound::Message(message)
                    if message.get("type").and_then(Value::as_str) == Some("result") =>
                {
                    return;
                }
                Inbound::Exited(_) => return,
                Inbound::ToolCall(call) => {
                    let cancelled = json!({"content": [{"type": "text", "text": "Cancelled by user."}], "isError": true});
                    let _ = live.process.answer_tool(&call, &cancelled).await;
                }
                Inbound::Message(_) => {}
            }
        }
    };
    if tokio::time::timeout(DRAIN_TIMEOUT, drain).await.is_err() {
        tracing::warn!("claude-agent: the interrupted turn did not finish in time");
    }
}
