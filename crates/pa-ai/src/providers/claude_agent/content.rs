//! Conversion between Prime Agent messages and what `claude` accepts: user content blocks,
//! the `<conversation-checkpoint>` for history it has not seen, MCP tool definitions, and MCP
//! tool-call results.

use std::collections::HashMap;

use serde_json::{json, Value};

use super::fingerprint::{groups, GroupRole};
use crate::types::{
    AssistantContent, Message, Tool, UserBlockPayload, UserMessageContent, UserOrToolContent,
};

/// The MCP server name Prime Agent's tools are offered under.
pub(crate) const SERVER_NAME: &str = "prime";
/// The prefix Claude Code gives every tool of [`SERVER_NAME`].
pub(crate) const TOOL_PREFIX: &str = "mcp__prime__";
/// Claude Code cuts each MCP tool description at this many characters.
const DESCRIPTION_LIMIT: usize = 2048;

const CHECKPOINT_OPEN: &str = "<conversation-checkpoint>\nThe following is conversation history from Prime Agent that you have not seen. Treat it as prior conversation, not a new request. Continue with the user messages after this checkpoint.";
const CHECKPOINT_CLOSE: &str = "</conversation-checkpoint>";

fn text_block(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

fn image_block(data: &str, mime_type: &str) -> Value {
    json!({"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": data}})
}

fn push_text(blocks: &mut Vec<Value>, text: &str) {
    // The API rejects empty text blocks.
    if !text.is_empty() {
        blocks.push(text_block(text));
    }
}

fn push_user_blocks(blocks: &mut Vec<Value>, content: &[UserOrToolContent]) {
    for block in content {
        match crate::types::user_block_payload(block) {
            UserBlockPayload::Text(text) => push_text(blocks, text),
            UserBlockPayload::Opaque(text) => push_text(blocks, &text),
            UserBlockPayload::Image { data, mime_type } => {
                blocks.push(image_block(data, mime_type));
            }
        }
    }
}

fn tool_output(content: &[UserOrToolContent]) -> Value {
    let parts: Vec<Value> = content
        .iter()
        .map(|block| match crate::types::user_block_payload(block) {
            UserBlockPayload::Text(text) => json!({"type": "text", "text": text}),
            UserBlockPayload::Opaque(text) => json!({"type": "text", "text": text}),
            UserBlockPayload::Image { mime_type, .. } => {
                json!({"type": "image", "mediaType": mime_type})
            }
        })
        .collect();
    Value::Array(parts)
}

/// Messages as one user turn. Claude cannot be handed earlier turns, so when the slice holds any
/// history its roles become text tags; images stay real content blocks.
pub(crate) fn content(messages: &[Message]) -> Vec<Value> {
    let history = messages
        .iter()
        .any(|message| matches!(message, Message::Assistant(_) | Message::ToolResult(_)));
    let mut blocks = Vec::new();
    for message in messages {
        let role = match message {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::ToolResult(_) => "tool",
        };
        if history {
            blocks.push(text_block(&format!("<{role}>")));
        }
        match message {
            Message::User(user) => match &user.content {
                UserMessageContent::Text(text) => push_text(&mut blocks, text),
                UserMessageContent::Blocks(content) => push_user_blocks(&mut blocks, content),
            },
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    match block {
                        AssistantContent::Text(text) => push_text(&mut blocks, &text.text),
                        AssistantContent::Thinking(_) => {}
                        AssistantContent::ToolCall(call) => {
                            let payload = json!({
                                "id": call.id,
                                "name": call.name,
                                "input": Value::Object(call.arguments.clone()),
                            });
                            blocks
                                .push(text_block(&format!("<tool-call>\n{payload}\n</tool-call>")));
                        }
                    }
                }
            }
            Message::ToolResult(result) => {
                let payload = json!({
                    "id": result.tool_call_id,
                    "name": result.tool_name,
                    "output": tool_output(&result.content),
                    "isError": result.is_error,
                });
                blocks.push(text_block(&format!(
                    "<tool-result>\n{payload}\n</tool-result>"
                )));
                for block in &result.content {
                    if let UserBlockPayload::Image { data, mime_type } =
                        crate::types::user_block_payload(block)
                    {
                        blocks.push(image_block(data, mime_type));
                    }
                }
            }
        }
        if history {
            blocks.push(text_block(&format!("</{role}>")));
        }
    }
    blocks
}

/// Everything from `start` on that `claude` has not seen: earlier history inside a checkpoint,
/// then the trailing user input as plain content.
pub(crate) fn catch_up(messages: &[Message], start: usize) -> Vec<Value> {
    let missed = &messages[start.min(messages.len())..];
    let end = groups(missed, &[])
        .iter()
        .rev()
        .find(|group| group.role != GroupRole::User)
        .map_or(0, |group| group.end);
    let mut blocks = Vec::new();
    if end > 0 {
        blocks.push(text_block(CHECKPOINT_OPEN));
        blocks.extend(content(&missed[..end]));
        blocks.push(text_block(CHECKPOINT_CLOSE));
    }
    blocks.extend(content(&missed[end..]));
    blocks
}

/// The MCP name Claude Code sees for a tool: names are capped at 64 characters with the prefix.
pub(crate) fn short_name(name: &str) -> String {
    let limit = 64 - TOOL_PREFIX.len();
    let valid = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if name.len() <= limit && valid {
        return name.to_string();
    }
    let digest = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(name.as_bytes()));
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(limit - 11)
        .collect();
    format!("{cleaned}_{}", &digest[..10])
}

/// The offered tool behind a Claude Code tool name; `None` for a tool we did not offer (Claude
/// Code answers those calls itself).
pub(crate) fn original_name<'a>(name: &str, tools: &'a [Tool]) -> Option<&'a str> {
    tools
        .iter()
        .find(|tool| format!("{TOOL_PREFIX}{}", short_name(&tool.name)) == name)
        .map(|tool| tool.name.as_str())
}

/// The `tools/list` result for the offered tools.
pub(crate) fn mcp_tools(tools: &[Tool]) -> Value {
    let list: Vec<Value> = tools
        .iter()
        .map(|tool| {
            let mut schema = tool.parameters.clone();
            if let Value::Object(map) = &mut schema {
                map.entry("type").or_insert_with(|| json!("object"));
            } else {
                schema = json!({"type": "object"});
            }
            let description: String = tool.description.chars().take(DESCRIPTION_LIMIT).collect();
            json!({
                "name": short_name(&tool.name),
                "description": description,
                "inputSchema": schema,
            })
        })
        .collect();
    json!({ "tools": list })
}

/// The MCP `tools/call` result for one Prime Agent tool result.
pub(crate) fn mcp_result(content: &[UserOrToolContent], is_error: bool) -> Value {
    let mut parts = Vec::new();
    for block in content {
        match crate::types::user_block_payload(block) {
            UserBlockPayload::Text(text) => parts.push(json!({"type": "text", "text": text})),
            UserBlockPayload::Opaque(text) => parts.push(json!({"type": "text", "text": text})),
            UserBlockPayload::Image { data, mime_type } => {
                parts.push(json!({"type": "image", "mimeType": mime_type, "data": data}));
            }
        }
    }
    json!({ "content": parts, "isError": is_error })
}

/// The tool results at `start`: one group of consecutive tool-result messages, keyed by call id,
/// plus the index after them.
pub(crate) fn collect_results(
    messages: &[Message],
    start: usize,
) -> (HashMap<String, Value>, usize) {
    let mut results = HashMap::new();
    let mut end = start;
    while let Some(Message::ToolResult(result)) = messages.get(end) {
        results.insert(
            result.tool_call_id.clone(),
            mcp_result(&result.content, result.is_error),
        );
        end += 1;
    }
    (results, end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TextContent, ToolResultMessage, UserMessage};
    use serde_json::Map;

    fn user(text: &str) -> Message {
        Message::User(UserMessage {
            content: UserMessageContent::Text(text.to_string()),
            timestamp: 0,
            rest: Map::default(),
        })
    }

    fn result(id: &str, text: &str) -> Message {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: id.to_string(),
            tool_name: "bash".to_string(),
            content: vec![UserOrToolContent::Text(TextContent {
                text: text.to_string(),
                text_signature: None,
                rest: Map::default(),
            })],
            details: None,
            is_error: false,
            timestamp: 0,
            rest: Map::default(),
        })
    }

    #[test]
    fn plain_user_input_has_no_role_tags() {
        assert_eq!(
            content(&[user("hi"), user("")]),
            vec![json!({"type": "text", "text": "hi"})]
        );
    }

    #[test]
    fn unseen_history_goes_inside_a_checkpoint_before_the_new_input() {
        let messages = vec![user("old"), result("t1", "out"), user("new")];
        let blocks = catch_up(&messages, 0);
        assert_eq!(blocks.first(), Some(&text_block(CHECKPOINT_OPEN)));
        assert_eq!(blocks[blocks.len() - 2], text_block(CHECKPOINT_CLOSE));
        assert_eq!(blocks.last(), Some(&text_block("new")));
    }

    #[test]
    fn tool_names_round_trip_through_the_prefix() {
        let tools: Vec<Tool> = vec![serde_json::from_value(
            json!({"name": "python_repl", "description": "d", "parameters": {}}),
        )
        .unwrap()];
        assert_eq!(
            original_name("mcp__prime__python_repl", &tools),
            Some("python_repl")
        );
        assert_eq!(original_name("Bash", &tools), None);
        assert_eq!(
            mcp_tools(&tools),
            json!({"tools": [{"name": "python_repl", "description": "d", "inputSchema": {"type": "object"}}]})
        );
    }

    #[test]
    fn consecutive_tool_results_are_collected_up_to_the_next_message() {
        let messages = vec![user("a"), result("t1", "x"), result("t2", "y"), user("b")];
        let (results, end) = collect_results(&messages, 1);
        assert_eq!(end, 3);
        assert_eq!(
            results.get("t2"),
            Some(&json!({"content": [{"type": "text", "text": "y"}], "isError": false}))
        );
    }
}
