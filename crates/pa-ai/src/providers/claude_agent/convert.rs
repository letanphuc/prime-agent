//! `claude` stream-json output to Prime Agent stream events: the forwarded Anthropic SSE events
//! drive text, thinking and tool-call blocks; complete assistant messages fill in what was not
//! streamed; the `result` message ends the turn or reports its failure.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use serde_json::{Map, Value};

use super::content::original_name;
use crate::event_stream::{AssistantMessageEvent, AssistantMessageEventWriter};
use crate::types::{
    AssistantContent, AssistantMessage, Model, StopReason, TextContent, ThinkingContent, Tool,
    ToolCall, Usage,
};
use crate::utils_inner::json_parse::parse_json_with_repair;

/// Where a read step ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// The reply called offered tools; the `claude` turn waits for their results.
    Tools,
    /// The `claude` turn finished.
    Finished,
}

#[derive(Default, Clone, Copy)]
struct Counts {
    input: u64,
    read: u64,
    write: u64,
    output: u64,
}

/// Claude Code reports one response's usage several times while it streams, and again as a turn
/// total. Keep the highest count per response; the last response's input is the context size.
#[derive(Default)]
struct UsageTracker {
    messages: HashMap<String, Counts>,
    order: Vec<String>,
}

impl UsageTracker {
    fn update(&mut self, id: &str, usage: &Value) {
        let get = |field: &str| usage.get(field).and_then(Value::as_u64).unwrap_or(0);
        if !self.messages.contains_key(id) {
            self.order.push(id.to_string());
        }
        let counts = self.messages.entry(id.to_string()).or_default();
        counts.input = counts.input.max(get("input_tokens"));
        counts.read = counts.read.max(get("cache_read_input_tokens"));
        counts.write = counts.write.max(get("cache_creation_input_tokens"));
        counts.output = counts.output.max(get("output_tokens"));
    }

    fn snapshot(&self) -> Usage {
        let last = self
            .order
            .last()
            .and_then(|id| self.messages.get(id))
            .copied()
            .unwrap_or_default();
        let output = self.messages.values().map(|counts| counts.output).sum();
        let mut usage = Usage {
            input: last.input,
            output,
            cache_read: last.read,
            cache_write: last.write,
            ..Usage::default()
        };
        usage.total_tokens = crate::types::usage_total_tokens(&usage);
        usage
    }
}

enum Block {
    Text(usize),
    Thinking(usize),
    Tool { position: usize, json: String },
    Ignored,
}

/// Per-request conversion state.
pub(crate) struct Converter<'a> {
    model: &'a Model,
    tools: &'a [Tool],
    /// A subscription login is covered by the plan, so its cost shows as $0.
    subscription: bool,
    pub output: AssistantMessage,
    blocks: HashMap<u64, Block>,
    message_id: String,
    streamed: HashSet<String>,
    called_tools: bool,
    usage: UsageTracker,
    error_code: Option<String>,
    rejected: Option<Value>,
}

fn is_main_thread(message: &Value) -> bool {
    message.get("parent_tool_use_id").is_none_or(Value::is_null)
}

fn str_field<'v>(value: &'v Value, field: &str) -> &'v str {
    value.get(field).and_then(Value::as_str).unwrap_or("")
}

impl<'a> Converter<'a> {
    pub(crate) fn new(model: &'a Model, tools: &'a [Tool], subscription: bool) -> Self {
        Self {
            model,
            tools,
            subscription,
            output: AssistantMessage {
                content: Vec::new(),
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                diagnostics: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                stop_reason_raw: None,
                error_message: None,
                timestamp: crate::utils_inner::diagnostics::now_ms(),
                rest: Map::default(),
            },
            blocks: HashMap::new(),
            message_id: String::new(),
            streamed: HashSet::new(),
            called_tools: false,
            usage: UsageTracker::default(),
            error_code: None,
            rejected: None,
        }
    }

    /// Price the usage as a subscription ($0) or at the model's list prices.
    pub(crate) fn set_subscription(&mut self, subscription: bool) {
        self.subscription = subscription;
    }

    /// Ids of the offered-tool calls in the reply so far.
    pub(crate) fn tool_call_ids(&self) -> Vec<String> {
        self.output
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::ToolCall(call) => Some(call.id.clone()),
                AssistantContent::Text(_) | AssistantContent::Thinking(_) => None,
            })
            .collect()
    }

    fn refresh_usage(&mut self) {
        self.output.usage = self.usage.snapshot();
        if self.subscription {
            self.output.usage.cost = crate::types::UsageCost::default();
        } else {
            crate::models::calculate_cost(self.model, &mut self.output.usage, None);
        }
    }

    fn push_block(&mut self, block: AssistantContent) -> usize {
        self.output.content.push(block);
        self.output.content.len() - 1
    }

    /// Handle one stream-json message. `Err` is the turn's failure text.
    pub(crate) fn handle(
        &mut self,
        message: &Value,
        writer: &AssistantMessageEventWriter,
    ) -> Result<Option<Outcome>, String> {
        match str_field(message, "type") {
            "rate_limit_event" => {
                let info = message.get("rate_limit_info").cloned().unwrap_or_default();
                self.rejected = (str_field(&info, "status") == "rejected").then_some(info);
                Ok(None)
            }
            "stream_event" if is_main_thread(message) => {
                let event = message.get("event").cloned().unwrap_or_default();
                Ok(self.handle_event(&event, writer))
            }
            "assistant" if is_main_thread(message) => {
                self.handle_assistant(message, writer);
                Ok(None)
            }
            "result" => self.handle_result(message).map(Some),
            _ => Ok(None),
        }
    }

    fn handle_event(
        &mut self,
        event: &Value,
        writer: &AssistantMessageEventWriter,
    ) -> Option<Outcome> {
        let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
        match str_field(event, "type") {
            "message_start" => {
                let message = event.get("message").cloned().unwrap_or_default();
                self.message_id = str_field(&message, "id").to_string();
                self.streamed.insert(self.message_id.clone());
                self.blocks.clear();
                self.output.response_id = Some(self.message_id.clone());
                self.output.response_model = message
                    .get("model")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                if let Some(usage) = message.get("usage") {
                    let id = self.message_id.clone();
                    self.usage.update(&id, usage);
                    self.refresh_usage();
                }
            }
            "content_block_start" => self.start_block(index, event, writer),
            "content_block_delta" => self.delta(index, event, writer),
            "content_block_stop" => self.stop_block(index, writer),
            "message_delta" => {
                if let Some(usage) = event.get("usage") {
                    let id = self.message_id.clone();
                    self.usage.update(&id, usage);
                    self.refresh_usage();
                }
                let reason = event
                    .get("delta")
                    .and_then(|delta| delta.get("stop_reason"))
                    .and_then(Value::as_str);
                if let Some(reason) = reason {
                    self.output.stop_reason_raw = Some(reason.to_string());
                }
            }
            "message_stop" if self.called_tools => {
                self.output.stop_reason = StopReason::ToolUse;
                return Some(Outcome::Tools);
            }
            _ => {}
        }
        None
    }

    fn start_block(&mut self, index: u64, event: &Value, writer: &AssistantMessageEventWriter) {
        let block = event.get("content_block").cloned().unwrap_or_default();
        match str_field(&block, "type") {
            "text" => {
                let position = self.push_block(AssistantContent::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                    rest: Map::default(),
                }));
                self.blocks.insert(index, Block::Text(position));
                writer.push(AssistantMessageEvent::TextStart {
                    content_index: position as u64,
                    partial: self.output.clone(),
                });
                let text = str_field(&block, "text").to_string();
                if !text.is_empty() {
                    self.append_text(position, &text, writer);
                }
            }
            "thinking" | "redacted_thinking" => {
                let redacted = str_field(&block, "type") == "redacted_thinking";
                let position = self.push_block(AssistantContent::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: redacted.then(|| str_field(&block, "data").to_string()),
                    redacted: redacted.then_some(true),
                    rest: Map::default(),
                }));
                self.blocks.insert(index, Block::Thinking(position));
                writer.push(AssistantMessageEvent::ThinkingStart {
                    content_index: position as u64,
                    partial: self.output.clone(),
                });
                let thinking = str_field(&block, "thinking").to_string();
                if !thinking.is_empty() {
                    self.append_thinking(position, &thinking, writer);
                }
            }
            "tool_use" => {
                let Some(name) = original_name(str_field(&block, "name"), self.tools) else {
                    // Claude Code answers calls to tools we did not offer itself.
                    self.blocks.insert(index, Block::Ignored);
                    return;
                };
                self.called_tools = true;
                let position = self.push_block(AssistantContent::ToolCall(ToolCall {
                    id: str_field(&block, "id").to_string(),
                    name: name.to_string(),
                    arguments: Map::new(),
                    thought_signature: None,
                    rest: Map::default(),
                }));
                self.blocks.insert(
                    index,
                    Block::Tool {
                        position,
                        json: String::new(),
                    },
                );
                writer.push(AssistantMessageEvent::ToolcallStart {
                    content_index: position as u64,
                    partial: self.output.clone(),
                });
            }
            _ => {
                self.blocks.insert(index, Block::Ignored);
            }
        }
    }

    fn append_text(&mut self, position: usize, delta: &str, writer: &AssistantMessageEventWriter) {
        if let Some(AssistantContent::Text(text)) = self.output.content.get_mut(position) {
            text.text.push_str(delta);
        }
        writer.push(AssistantMessageEvent::TextDelta {
            content_index: position as u64,
            delta: delta.to_string(),
            partial: self.output.clone(),
        });
    }

    fn append_thinking(
        &mut self,
        position: usize,
        delta: &str,
        writer: &AssistantMessageEventWriter,
    ) {
        if let Some(AssistantContent::Thinking(thinking)) = self.output.content.get_mut(position) {
            thinking.thinking.push_str(delta);
        }
        writer.push(AssistantMessageEvent::ThinkingDelta {
            content_index: position as u64,
            delta: delta.to_string(),
            partial: self.output.clone(),
        });
    }

    fn delta(&mut self, index: u64, event: &Value, writer: &AssistantMessageEventWriter) {
        let delta = event.get("delta").cloned().unwrap_or_default();
        match self.blocks.get_mut(&index) {
            Some(Block::Text(position)) => {
                let position = *position;
                let text = str_field(&delta, "text").to_string();
                if !text.is_empty() {
                    self.append_text(position, &text, writer);
                }
            }
            Some(Block::Thinking(position)) => {
                let position = *position;
                if str_field(&delta, "type") == "signature_delta" {
                    if let Some(AssistantContent::Thinking(thinking)) =
                        self.output.content.get_mut(position)
                    {
                        thinking.thinking_signature =
                            Some(str_field(&delta, "signature").to_string());
                    }
                } else {
                    let thinking = str_field(&delta, "thinking").to_string();
                    if !thinking.is_empty() {
                        self.append_thinking(position, &thinking, writer);
                    }
                }
            }
            Some(Block::Tool { position, json }) => {
                let partial = str_field(&delta, "partial_json");
                json.push_str(partial);
                let position = *position;
                writer.push(AssistantMessageEvent::ToolcallDelta {
                    content_index: position as u64,
                    delta: partial.to_string(),
                    partial: self.output.clone(),
                });
            }
            Some(Block::Ignored) | None => {}
        }
    }

    fn stop_block(&mut self, index: u64, writer: &AssistantMessageEventWriter) {
        match self.blocks.remove(&index) {
            Some(Block::Text(position)) => {
                let content = match self.output.content.get(position) {
                    Some(AssistantContent::Text(text)) => text.text.clone(),
                    _ => String::new(),
                };
                writer.push(AssistantMessageEvent::TextEnd {
                    content_index: position as u64,
                    content,
                    partial: self.output.clone(),
                });
            }
            Some(Block::Thinking(position)) => {
                let content = match self.output.content.get(position) {
                    Some(AssistantContent::Thinking(thinking)) => thinking.thinking.clone(),
                    _ => String::new(),
                };
                writer.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: position as u64,
                    content,
                    partial: self.output.clone(),
                });
            }
            Some(Block::Tool { position, json }) => {
                let arguments = if json.trim().is_empty() {
                    Map::new()
                } else {
                    parse_json_with_repair(&json)
                        .ok()
                        .and_then(|value| value.as_object().cloned())
                        .unwrap_or_default()
                };
                let Some(AssistantContent::ToolCall(call)) = self.output.content.get_mut(position)
                else {
                    return;
                };
                call.arguments = arguments;
                let tool_call = call.clone();
                writer.push(AssistantMessageEvent::ToolcallEnd {
                    content_index: position as u64,
                    tool_call,
                    partial: self.output.clone(),
                });
            }
            Some(Block::Ignored) | None => {}
        }
    }

    fn handle_assistant(&mut self, message: &Value, writer: &AssistantMessageEventWriter) {
        if let Some(code) = message.get("error").and_then(Value::as_str) {
            // Synthetic API errors are diagnostics, not answers; the result reports them.
            self.error_code = Some(code.to_string());
            return;
        }
        let body = message.get("message").cloned().unwrap_or_default();
        let id = str_field(&body, "id").to_string();
        if let Some(usage) = body.get("usage") {
            self.usage.update(&id, usage);
            self.refresh_usage();
        }
        let blocks = body
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let streamed = self.streamed.contains(&id);
        for block in &blocks {
            match str_field(block, "type") {
                "tool_use" => {
                    let Some(name) = original_name(str_field(block, "name"), self.tools) else {
                        continue;
                    };
                    self.called_tools = true;
                    let call_id = str_field(block, "id");
                    let arguments = block
                        .get("input")
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default();
                    let existing = self.output.content.iter_mut().find_map(|item| match item {
                        AssistantContent::ToolCall(call) if call.id == call_id => Some(call),
                        _ => None,
                    });
                    // The complete message carries the final input of a streamed call.
                    if let Some(call) = existing {
                        call.arguments = arguments;
                        continue;
                    }
                    let call = ToolCall {
                        id: call_id.to_string(),
                        name: name.to_string(),
                        arguments,
                        thought_signature: None,
                        rest: Map::default(),
                    };
                    let position = self.push_block(AssistantContent::ToolCall(call.clone()));
                    writer.push(AssistantMessageEvent::ToolcallStart {
                        content_index: position as u64,
                        partial: self.output.clone(),
                    });
                    writer.push(AssistantMessageEvent::ToolcallEnd {
                        content_index: position as u64,
                        tool_call: call,
                        partial: self.output.clone(),
                    });
                }
                // Complete messages repeat streamed content; use them only when nothing streamed.
                "text" if !streamed => {
                    let position = self.push_block(AssistantContent::Text(TextContent {
                        text: String::new(),
                        text_signature: None,
                        rest: Map::default(),
                    }));
                    writer.push(AssistantMessageEvent::TextStart {
                        content_index: position as u64,
                        partial: self.output.clone(),
                    });
                    self.append_text(position, str_field(block, "text"), writer);
                    self.blocks.insert(u64::MAX, Block::Text(position));
                    self.stop_block(u64::MAX, writer);
                }
                "thinking" if !streamed => {
                    let position = self.push_block(AssistantContent::Thinking(ThinkingContent {
                        thinking: String::new(),
                        thinking_signature: block
                            .get("signature")
                            .and_then(Value::as_str)
                            .map(ToString::to_string),
                        redacted: None,
                        rest: Map::default(),
                    }));
                    writer.push(AssistantMessageEvent::ThinkingStart {
                        content_index: position as u64,
                        partial: self.output.clone(),
                    });
                    self.append_thinking(position, str_field(block, "thinking"), writer);
                    self.blocks.insert(u64::MAX, Block::Thinking(position));
                    self.stop_block(u64::MAX, writer);
                }
                _ => {}
            }
        }
    }

    fn handle_result(&mut self, message: &Value) -> Result<Outcome, String> {
        let success = str_field(message, "subtype") == "success";
        let is_error = message
            .get("is_error")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let status = message.get("api_error_status").and_then(Value::as_u64);
        let http_error = status.is_some_and(|status| status >= 400);
        if success && !is_error && !http_error {
            if message.get("num_turns").and_then(Value::as_u64) == Some(1)
                && self.usage.order.len() <= 1
            {
                let id = self
                    .usage
                    .order
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "result".to_string());
                if let Some(usage) = message.get("usage") {
                    self.usage.update(&id, usage);
                }
            }
            self.refresh_usage();
            if str_field(message, "stop_reason") == "max_tokens" {
                self.output.stop_reason = StopReason::Length;
            }
            self.output.stop_reason_raw = message
                .get("stop_reason")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            return Ok(Outcome::Finished);
        }
        Err(self.failure_text(message, success, status))
    }

    fn failure_text(&self, message: &Value, success: bool, status: Option<u64>) -> String {
        let result = if success {
            str_field(message, "result").to_string()
        } else {
            message
                .get("errors")
                .and_then(Value::as_array)
                .map(|errors| {
                    errors
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default()
        }
        .replace("Please run /login", "Run `claude auth login` in a terminal");
        let failure = match status {
            Some(529) => {
                "Anthropic is overloaded (529). Claude Code exhausted its retries. Try again shortly."
                    .to_string()
            }
            Some(status) if status >= 500 => format!(
                "Anthropic returned HTTP {status}. Claude Code exhausted its retries. Try again shortly."
            ),
            _ => result,
        };
        let limited = self.error_code.as_deref() == Some("rate_limit")
            || (self.rejected.is_some() && status == Some(429));
        let window = self
            .rejected
            .as_ref()
            .and_then(|info| info.get("rateLimitType"))
            .and_then(Value::as_str)
            .map(|kind| kind.replace('_', " "));
        let mut text = if limited && failure.is_empty() {
            match window {
                Some(window) => format!("Claude Code subscription limit reached ({window})."),
                None => "Claude Code subscription limit reached.".to_string(),
            }
        } else {
            failure
        };
        if limited {
            if let Some(resets) = self
                .rejected
                .as_ref()
                .and_then(|info| info.get("resetsAt"))
                .and_then(Value::as_u64)
            {
                let _ = write!(text, " Resets at unix time {resets}.");
            }
        }
        if self.error_code.as_deref() == Some("billing_error") {
            text.push_str(" Check your Claude plan and extra-usage settings.");
        }
        if !text.is_empty() {
            return text;
        }
        match status {
            Some(status) => format!("Claude Code could not finish its reply (HTTP {status})"),
            None => "Claude Code could not finish its reply".to_string(),
        }
    }
}
