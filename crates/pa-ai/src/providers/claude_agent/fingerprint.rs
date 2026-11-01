//! History fingerprints: which part of the conversation the running `claude` session has already
//! seen. Messages fold into groups (consecutive user messages, one assistant reply, consecutive
//! tool results); each group hashes its visible content, so thinking text and provider metadata
//! never decide whether two histories match.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::types::{AssistantContent, Message, UserMessageContent, UserOrToolContent};

/// Role of a history group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GroupRole {
    User,
    Assistant,
    Tool,
}

/// One history group: its hash and the message range `start..end` it covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Group {
    pub hash: String,
    pub role: GroupRole,
    pub start: usize,
    pub end: usize,
}

/// What `claude` has seen of a session: the group hashes through its last reply.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SessionRecord {
    pub claude_session_id: String,
    pub history: Vec<String>,
    /// After a cancelled turn `claude` owns the partial reply; history ends at the input it got.
    pub interrupted: bool,
}

/// How the incoming history relates to the record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// Nothing matches: start a new `claude` session with the whole history.
    Fresh,
    /// An earlier part changed (edit, compaction): start over with the whole history.
    Changed,
    /// Only new user input after the record: send it as is.
    Continue { start: usize },
    /// Unseen history that is not only user input: send it inside a checkpoint.
    CatchUp { start: usize },
}

/// Serialize with sorted object keys, so equal content hashes equally whatever the map order.
fn canonical(value: &Value) -> String {
    match value {
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(canonical).collect();
            format!("[{}]", parts.join(","))
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .into_iter()
                .map(|key| format!("{}:{}", Value::String(key.clone()), canonical(&map[key])))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        other => other.to_string(),
    }
}

fn hash(value: &Value) -> String {
    let digest = Sha256::digest(canonical(value).as_bytes());
    hex::encode(digest)
}

fn block_parts(blocks: &[UserOrToolContent], parts: &mut Vec<Value>, text: &mut String) {
    for block in blocks {
        match crate::types::user_block_payload(block) {
            crate::types::UserBlockPayload::Text(value) => text.push_str(value),
            crate::types::UserBlockPayload::Opaque(value) => text.push_str(&value),
            crate::types::UserBlockPayload::Image { data, mime_type } => {
                flush_text(parts, text);
                parts.push(json!({"type": "file", "mediaType": mime_type, "data": data}));
            }
        }
    }
}

fn flush_text(parts: &mut Vec<Value>, text: &mut String) {
    if !text.is_empty() {
        parts.push(json!({"type": "text", "text": std::mem::take(text)}));
    }
}

/// The content that decides whether two messages match.
pub(crate) fn visible(message: &Message) -> Vec<Value> {
    let mut parts = Vec::new();
    let mut text = String::new();
    match message {
        Message::User(user) => match &user.content {
            UserMessageContent::Text(value) => text.push_str(value),
            UserMessageContent::Blocks(blocks) => block_parts(blocks, &mut parts, &mut text),
        },
        Message::Assistant(assistant) => {
            for block in &assistant.content {
                match block {
                    AssistantContent::Text(value) => text.push_str(&value.text),
                    AssistantContent::Thinking(_) => {}
                    AssistantContent::ToolCall(call) => {
                        flush_text(&mut parts, &mut text);
                        parts.push(json!({
                            "type": "tool-call",
                            "id": call.id,
                            "name": call.name,
                            "input": Value::Object(call.arguments.clone()),
                        }));
                    }
                }
            }
        }
        Message::ToolResult(result) => {
            let mut output = Vec::new();
            let mut output_text = String::new();
            block_parts(&result.content, &mut output, &mut output_text);
            flush_text(&mut output, &mut output_text);
            parts.push(json!({
                "type": "tool-result",
                "id": result.tool_call_id,
                "name": result.tool_name,
                "output": output,
                "isError": result.is_error,
            }));
        }
    }
    flush_text(&mut parts, &mut text);
    parts
}

fn role_of(message: &Message) -> GroupRole {
    match message {
        Message::User(_) => GroupRole::User,
        Message::Assistant(_) => GroupRole::Assistant,
        Message::ToolResult(_) => GroupRole::Tool,
    }
}

fn role_name(role: GroupRole) -> &'static str {
    match role {
        GroupRole::User => "user",
        GroupRole::Assistant => "assistant",
        GroupRole::Tool => "tool",
    }
}

fn group_hash(role: GroupRole, members: &[String]) -> String {
    hash(&json!({"role": role_name(role), "members": members}))
}

/// Fold `messages` into groups. `prefix` is the saved history: a submitted user input stays a
/// boundary even when more user messages follow it without a reply in between.
pub(crate) fn groups(messages: &[Message], prefix: &[String]) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        let start = index;
        index += 1;
        let parts = visible(message);
        if parts.is_empty() {
            continue;
        }
        let role = role_of(message);
        let mut members = vec![hash(&Value::Array(parts))];
        match role {
            GroupRole::User => {
                while index < messages.len() {
                    if let Some(checkpoint) = prefix.get(groups.len()) {
                        let mut sorted = members.clone();
                        sorted.sort();
                        if group_hash(role, &sorted) == *checkpoint {
                            break;
                        }
                    }
                    let next = &messages[index];
                    if role_of(next) != GroupRole::User {
                        break;
                    }
                    members.push(hash(&Value::Array(visible(next))));
                    index += 1;
                }
                members.sort();
            }
            GroupRole::Tool => {
                while index < messages.len() && role_of(&messages[index]) == GroupRole::Tool {
                    members.push(hash(&Value::Array(visible(&messages[index]))));
                    index += 1;
                }
            }
            GroupRole::Assistant => {}
        }
        groups.push(Group {
            hash: group_hash(role, &members),
            role,
            start,
            end: index,
        });
    }
    groups
}

/// Compare the incoming history with what `claude` has seen.
pub(crate) fn decide(messages: &[Message], record: Option<&SessionRecord>) -> Decision {
    let saved: &[String] = record.map_or(&[], |record| &record.history);
    let interrupted = record.is_some_and(|record| record.interrupted);
    let groups = groups(messages, if interrupted { saved } else { &[] });
    if saved.is_empty() {
        return Decision::Fresh;
    }
    let mut matched = 0;
    while matched < saved.len() && matched < groups.len() && saved[matched] == groups[matched].hash
    {
        matched += 1;
    }
    if matched == 0 {
        return Decision::Fresh;
    }
    if matched < saved.len() {
        return Decision::Changed;
    }
    // After a cancel, Claude's own transcript owns the interrupted reply, not our partial copy.
    if interrupted {
        while let Some(group) = groups.get(matched) {
            let Message::Assistant(reply) = &messages[group.start] else {
                break;
            };
            if reply
                .content
                .iter()
                .any(|block| matches!(block, AssistantContent::ToolCall(_)))
            {
                break;
            }
            matched += 1;
        }
    }
    let start = groups
        .get(matched)
        .map_or(messages.len(), |group| group.start);
    if interrupted && start == messages.len() {
        return Decision::Changed;
    }
    if groups[matched..]
        .iter()
        .any(|group| group.role != GroupRole::User)
    {
        Decision::CatchUp { start }
    } else {
        Decision::Continue { start }
    }
}

/// The record after a turn: every group of the request plus the reply `claude` just wrote.
pub(crate) fn checkpoints(messages: &[Message], reply: Option<&Message>) -> Vec<String> {
    let mut history: Vec<String> = groups(messages, &[])
        .into_iter()
        .map(|group| group.hash)
        .collect();
    if let Some(reply) = reply {
        if let Some(group) = groups(std::slice::from_ref(reply), &[]).into_iter().next() {
            history.push(group.hash);
        }
    }
    history
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AssistantMessage, StopReason, TextContent, ThinkingContent, ToolCall, ToolResultMessage,
        Usage, UserMessage,
    };
    use serde_json::Map;

    fn user(text: &str) -> Message {
        Message::User(UserMessage {
            content: UserMessageContent::Text(text.to_string()),
            timestamp: 0,
            rest: Map::default(),
        })
    }

    fn assistant(content: Vec<AssistantContent>) -> Message {
        Message::Assistant(AssistantMessage {
            content,
            api: "claude-agent".into(),
            provider: "claude-agent".into(),
            model: "m".into(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            stop_reason_raw: None,
            error_message: None,
            timestamp: 0,
            rest: Map::default(),
        })
    }

    fn text(value: &str) -> AssistantContent {
        AssistantContent::Text(TextContent {
            text: value.to_string(),
            text_signature: None,
            rest: Map::default(),
        })
    }

    fn call(id: &str) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall {
            id: id.to_string(),
            name: "bash".to_string(),
            arguments: json!({"command": "ls"}).as_object().cloned().unwrap(),
            thought_signature: None,
            rest: Map::default(),
        })
    }

    fn result(id: &str) -> Message {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: id.to_string(),
            tool_name: "bash".to_string(),
            content: vec![UserOrToolContent::Text(TextContent {
                text: "out".to_string(),
                text_signature: None,
                rest: Map::default(),
            })],
            details: None,
            is_error: false,
            timestamp: 0,
            rest: Map::default(),
        })
    }

    fn record(messages: &[Message], reply: &Message) -> SessionRecord {
        SessionRecord {
            claude_session_id: "s".to_string(),
            history: checkpoints(messages, Some(reply)),
            interrupted: false,
        }
    }

    #[test]
    fn a_new_user_message_after_the_reply_continues() {
        let first = vec![user("hi")];
        let reply = assistant(vec![text("hello")]);
        let saved = record(&first, &reply);
        let next = vec![user("hi"), reply, user("again")];
        assert_eq!(decide(&next, Some(&saved)), Decision::Continue { start: 2 });
    }

    #[test]
    fn tool_results_after_the_reply_catch_up() {
        let first = vec![user("run it")];
        let reply = assistant(vec![text("ok"), call("t1")]);
        let saved = record(&first, &reply);
        let next = vec![user("run it"), reply, result("t1")];
        assert_eq!(decide(&next, Some(&saved)), Decision::CatchUp { start: 2 });
    }

    #[test]
    fn thinking_does_not_change_the_fingerprint() {
        let first = vec![user("hi")];
        let reply = assistant(vec![text("hello")]);
        let saved = record(&first, &reply);
        let stored_reply = assistant(vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "hmm".to_string(),
                thinking_signature: Some("sig".to_string()),
                redacted: None,
                rest: Map::default(),
            }),
            text("hello"),
        ]);
        let next = vec![user("hi"), stored_reply, user("more")];
        assert_eq!(decide(&next, Some(&saved)), Decision::Continue { start: 2 });
    }

    #[test]
    fn an_edited_earlier_message_is_a_change_and_no_record_is_fresh() {
        let first = vec![user("hi"), assistant(vec![text("a")]), user("b")];
        let reply = assistant(vec![text("c")]);
        let saved = record(&first, &reply);
        let edited = vec![user("hi"), assistant(vec![text("a")]), user("B"), reply];
        assert_eq!(decide(&edited, Some(&saved)), Decision::Changed);
        assert_eq!(decide(&edited, None), Decision::Fresh);
        let replaced = vec![user("other")];
        assert_eq!(decide(&replaced, Some(&saved)), Decision::Fresh);
    }

    #[test]
    fn canonical_json_ignores_key_order() {
        let mut first = Map::new();
        first.insert("b".into(), json!(1));
        first.insert("a".into(), json!(2));
        let mut second = Map::new();
        second.insert("a".into(), json!(2));
        second.insert("b".into(), json!(1));
        assert_eq!(hash(&Value::Object(first)), hash(&Value::Object(second)));
    }
}
