//! End-to-end: a fake `claude` that speaks the stream-json protocol the way the real one did in a
//! recorded session (initialize, a streamed `tool_use`, the `tools/call` stamped with
//! `claudecode/toolUseId`, then a text reply and a `result`).

use serde_json::{json, Map};

use super::*;
use crate::types::{
    AssistantContent, Message, TextContent, Tool, ToolResultMessage, UserMessage,
    UserMessageContent, UserOrToolContent,
};

const FAKE_CLAUDE: &str = r#"#!/usr/bin/env python3
import json, os, sys

args = sys.argv[1:]
with open(os.environ["FAKE_CLAUDE_LOG"], "a") as log:
    log.write(json.dumps(args) + "\n")

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def event(ev):
    send({"type": "stream_event", "event": ev, "parent_tool_use_id": None, "session_id": "s"})

def read():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line)

def answer(msg):
    return msg["response"]["response"]["mcp_response"]["result"]

turn = 0
pid = str(os.getpid())
while True:
    msg = read()
    if msg["type"] == "control_request" and msg["request"]["subtype"] == "initialize":
        send({"type": "control_response", "response": {"subtype": "success", "request_id": msg["request_id"],
              "response": {"claude_code_version": "2.1.316", "account": {"subscriptionType": "Claude Max"}}}})
        continue
    if msg["type"] != "user":
        continue
    turn += 1
    text = "".join(b.get("text", "") for b in msg["message"]["content"])
    if turn == 1:
        event({"type": "message_start", "message": {"id": "msg_1", "model": "claude-test", "usage": {"input_tokens": 10, "output_tokens": 1}}})
        event({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "mcp__prime__bash", "input": {}}})
        event({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"ls\"}"}})
        event({"type": "content_block_stop", "index": 0})
        event({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}})
        event({"type": "message_stop"})
        send({"type": "control_request", "request_id": "call-1", "request": {"subtype": "mcp_message", "server_name": "prime",
              "message": {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "bash", "arguments": {"command": "ls"}, "_meta": {"claudecode/toolUseId": "toolu_1"}}}}})
        reply = read()
        while reply.get("type") != "control_response" or reply["response"]["request_id"] != "call-1":
            reply = read()
        tool_text = answer(reply)["content"][0]["text"]
        reply_text = "done: " + tool_text
    else:
        reply_text = "pid " + pid + " heard: " + text
    event({"type": "message_start", "message": {"id": "msg_%d" % (turn + 1), "model": "claude-test", "usage": {"input_tokens": 20}}})
    event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
    event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": reply_text}})
    event({"type": "content_block_stop", "index": 0})
    event({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}})
    event({"type": "message_stop"})
    send({"type": "result", "subtype": "success", "is_error": False, "stop_reason": "end_turn", "num_turns": 2, "result": reply_text})
"#;

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: UserMessageContent::Text(text.to_string()),
        timestamp: 0,
        rest: Map::default(),
    })
}

fn tool_result(id: &str, text: &str) -> Message {
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

fn model() -> Model {
    serde_json::from_value(json!({
        "id": "claude-test", "name": "Claude Test", "api": API_CLAUDE_AGENT, "provider": CLAUDE_AGENT_PROVIDER_ID,
        "baseUrl": LOCAL_BASE_URL, "reasoning": true, "input": ["text", "image"],
        "cost": {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 3.75},
        "contextWindow": 200_000, "maxTokens": 64_000,
    }))
    .expect("model parses")
}

fn text_of(message: &AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            AssistantContent::Thinking(_) | AssistantContent::ToolCall(_) => None,
        })
        .collect()
}

async fn complete(context: &Context, session_id: &str) -> AssistantMessage {
    let options = SimpleStreamOptions::from_base(StreamOptions {
        session_id: Some(session_id.to_string()),
        ..StreamOptions::default()
    });
    stream_claude_agent(&model(), context, Some(&options))
        .result()
        .await
}

// One test: the executable comes from a process-wide environment variable.
#[tokio::test]
async fn a_tool_call_parks_the_claude_turn_until_its_result_arrives() {
    let dir = tempfile::tempdir().expect("temp dir");
    let fake = dir.path().join("claude");
    std::fs::write(&fake, FAKE_CLAUDE).expect("write fake claude");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755))
            .expect("make fake claude executable");
    }
    let log = dir.path().join("args.log");
    std::env::set_var(CLAUDE_PATH_ENV, &fake);
    std::env::set_var("FAKE_CLAUDE_LOG", &log);

    let tools: Vec<Tool> = vec![serde_json::from_value(json!({
        "name": "bash", "description": "Run a command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    }))
    .expect("tool parses")];
    let mut context = Context {
        system_prompt: Some("You are Prime Agent.".to_string()),
        messages: vec![user("list the files")],
        tools: Some(tools),
    };

    // Step 1: Claude calls the offered tool; the stream ends at the tool boundary.
    let first = complete(&context, "session-a").await;
    assert_eq!(first.stop_reason, StopReason::ToolUse, "{first:?}");
    let calls: Vec<(String, String, serde_json::Value)> = first
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::ToolCall(call) => Some((
                call.id.clone(),
                call.name.clone(),
                serde_json::Value::Object(call.arguments.clone()),
            )),
            AssistantContent::Text(_) | AssistantContent::Thinking(_) => None,
        })
        .collect();
    assert_eq!(
        calls,
        vec![(
            "toolu_1".to_string(),
            "bash".to_string(),
            json!({"command": "ls"})
        )]
    );
    // A subscription login is covered by the plan.
    assert_eq!(first.usage.cost, crate::types::UsageCost::default());

    // Step 2: the tool result answers the parked `tools/call`; the same Claude turn finishes.
    context.messages.push(Message::Assistant(first));
    context.messages.push(tool_result("toolu_1", "a.txt b.txt"));
    let second = complete(&context, "session-a").await;
    assert_eq!(second.stop_reason, StopReason::Stop, "{second:?}");
    assert_eq!(text_of(&second), "done: a.txt b.txt");

    // Step 3: a new user message continues the same process with only the new input.
    context.messages.push(Message::Assistant(second));
    context.messages.push(user("thanks"));
    let third = complete(&context, "session-a").await;
    assert!(
        text_of(&third).ends_with(" heard: thanks"),
        "{}",
        text_of(&third)
    );

    let launches: Vec<Vec<String>> = std::fs::read_to_string(&log)
        .expect("args log")
        .lines()
        .map(|line| serde_json::from_str(line).expect("args line"))
        .collect();
    assert_eq!(launches.len(), 1, "one process serves the whole session");
    let args = &launches[0];
    let flag = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|index| args.get(index + 1))
            .cloned()
    };
    assert_eq!(flag("--model").as_deref(), Some("claude-test"));
    assert_eq!(flag("--tools").as_deref(), Some(""));
    assert_eq!(flag("--allowedTools").as_deref(), Some("mcp__prime__*"));
    assert_eq!(flag("--permission-mode").as_deref(), Some("dontAsk"));
    assert!(flag("--session-id").is_some());

    // A model the catalog derives keeps the Anthropic id and moves to this provider.
    let derived = claude_agent_model(&serde_json::from_value(json!({
        "id": "claude-opus-5-5", "name": "Claude Opus 5.5", "api": "anthropic-messages", "provider": "anthropic",
        "baseUrl": "https://api.anthropic.com", "reasoning": true, "input": ["text"],
        "cost": {"input": 5, "output": 25, "cacheRead": 0.5, "cacheWrite": 6.25},
        "contextWindow": 1_000_000, "maxTokens": 128_000,
    }))
    .expect("model parses"));
    assert_eq!(
        (
            derived.id.as_str(),
            derived.provider.as_str(),
            derived.api.as_str()
        ),
        (
            "claude-opus-5-5",
            CLAUDE_AGENT_PROVIDER_ID,
            API_CLAUDE_AGENT
        )
    );
}
