//! One `claude` child process speaking stream-json on stdin/stdout: launch flags, the
//! initialize handshake, control requests, user messages, and the in-process MCP server that
//! offers Prime Agent's tools. A reader task owns stdout; it answers the MCP handshake itself and
//! hands everything else to the session through [`Inbound`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, Mutex};

use super::content::{mcp_tools, SERVER_NAME, TOOL_PREFIX};
use crate::types::Tool;

/// Claude Code versions before this ignore `client_composed` and may read user text as commands.
const MINIMUM_VERSION: [u64; 3] = [2, 1, 248];
const START_TIMEOUT: Duration = Duration::from_secs(30);
const STOP_TIMEOUT: Duration = Duration::from_secs(5);
const STDERR_TAIL: usize = 4_000;

/// Claude Code features that duplicate Prime Agent's, turned off so Prime Agent's win.
const ISOLATION_ENV: [(&str, &str); 7] = [
    ("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "1"),
    ("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1"),
    ("DISABLE_AUTO_COMPACT", "1"),
    ("CLAUDE_CODE_ENABLE_AWAY_SUMMARY", "0"),
    ("ENABLE_TOOL_SEARCH", "false"),
    ("CLAUDE_CODE_DISABLE_ADVISOR_TOOL", "1"),
    // Raised so Prime Agent's own tool-output truncation applies instead.
    ("MAX_MCP_OUTPUT_TOKENS", "200000"),
];
/// Shell variables that would undo a launch setting: `CLAUDE_CODE_SIMPLE` skips the subscription
/// login, `CLAUDE_CODE_EFFORT_LEVEL` overrides the chosen effort, `MCP_TOOL_TIMEOUT` caps every
/// Prime Agent tool call.
const CLEARED_ENV: [&str; 3] = [
    "CLAUDE_CODE_SIMPLE",
    "CLAUDE_CODE_EFFORT_LEVEL",
    "MCP_TOOL_TIMEOUT",
];

/// How the system prompt reaches Claude.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SystemPrompt {
    /// Claude Code's own prompt with Prime Agent's appended (chat turns).
    Append(String),
    /// Prime Agent's prompt instead of Claude Code's (standalone side calls).
    Replace(String),
}

/// Whether Claude Code saves the conversation, and which one it continues.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Persistence {
    New(String),
    Resume(String),
    Discard,
}

/// Everything a launch needs.
pub(crate) struct Launch {
    pub executable: PathBuf,
    pub model: String,
    pub effort: Option<&'static str>,
    pub system: SystemPrompt,
    pub tools: Vec<Tool>,
    pub persistence: Persistence,
}

/// A `tools/call` from Claude Code, waiting for Prime Agent's result.
#[derive(Debug, Clone)]
pub(crate) struct ToolCallRequest {
    /// The control request to answer.
    control: String,
    /// The JSON-RPC id inside it.
    rpc: Value,
    /// Claude Code stamps each call with its `tool_use` id (`claudecode/toolUseId`).
    pub tool_use: Option<String>,
}

/// What the reader hands to the session.
#[derive(Debug)]
pub(crate) enum Inbound {
    Message(Value),
    ToolCall(ToolCallRequest),
    Exited(String),
}

type Controls = Arc<std::sync::Mutex<HashMap<String, oneshot::Sender<Result<Value, String>>>>>;

/// A running `claude`.
pub(crate) struct ClaudeProcess {
    child: Child,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    pub inbound: mpsc::UnboundedReceiver<Inbound>,
    controls: Controls,
    ended: Arc<AtomicBool>,
    next_request: AtomicU64,
    /// The Claude session id it saves under; `None` when nothing is saved.
    pub session_id: Option<String>,
    /// The initialize response: version, account, models.
    pub init: Value,
}

async fn write_line(stdin: &Mutex<Option<ChildStdin>>, value: &Value) -> Result<(), String> {
    let mut guard = stdin.lock().await;
    let Some(stdin) = guard.as_mut() else {
        return Err("Claude Code input is closed".to_string());
    };
    let mut line = value.to_string();
    line.push('\n');
    stdin
        .write_all(line.as_bytes())
        .await
        .map_err(|error| format!("Writing to Claude Code failed: {error}"))?;
    stdin
        .flush()
        .await
        .map_err(|error| format!("Writing to Claude Code failed: {error}"))
}

fn control_success(request_id: &str, response: &Value) -> Value {
    json!({"type": "control_response", "response": {"subtype": "success", "request_id": request_id, "response": response}})
}

fn mcp_reply(id: &Value, result: &Value) -> Value {
    json!({"mcp_response": {"jsonrpc": "2.0", "id": id, "result": result}})
}

/// A random version-4 UUID (Claude Code requires one for `--session-id`).
pub(crate) fn new_session_id() -> String {
    let mut bytes: [u8; 16] = rand::random();
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex = hex::encode(bytes);
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

fn version_error(version: Option<&str>) -> Option<String> {
    let Some(version) = version else {
        return Some(
            "Claude Code did not report its version, so compatibility could not be checked. Run `claude update`."
                .to_string(),
        );
    };
    let release = version.split(['-', '+']).next().unwrap_or(version);
    let parts: Vec<u64> = release
        .split('.')
        .map(|part| part.parse().unwrap_or(0))
        .collect();
    for (position, required) in MINIMUM_VERSION.iter().enumerate() {
        let current = parts.get(position).copied().unwrap_or(0);
        if current > *required {
            return None;
        }
        if current < *required {
            let minimum = MINIMUM_VERSION.map(|part| part.to_string()).join(".");
            return Some(format!(
                "Claude Code {version} is too old; {minimum} or later is required. Run `claude update`."
            ));
        }
    }
    None
}

fn launch_args(launch: &Launch) -> Vec<String> {
    let mut args: Vec<String> = [
        "--output-format",
        "stream-json",
        "--verbose",
        "--input-format",
        "stream-json",
        "--include-partial-messages",
        // Without summaries there is no reasoning text to show.
        "--thinking=adaptive",
        "--thinking-display",
        "summarized",
        "--model",
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    args.push(launch.model.clone());
    if let Some(effort) = launch.effort {
        args.extend(["--effort".to_string(), effort.to_string()]);
    }
    // Claude Code's built-in tools are off; only Prime Agent's tools reach Claude.
    args.extend(["--tools".to_string(), String::new()]);
    if !launch.tools.is_empty() {
        args.extend(["--allowedTools".to_string(), format!("{TOOL_PREFIX}*")]);
    }
    args.extend(
        [
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--permission-mode",
            "dontAsk",
            "--settings",
            r#"{"disableAllHooks":true}"#,
        ]
        .iter()
        .map(ToString::to_string),
    );
    match &launch.persistence {
        Persistence::New(id) => args.extend(["--session-id".to_string(), id.clone()]),
        Persistence::Resume(id) => args.extend(["--resume".to_string(), id.clone()]),
        Persistence::Discard => args.push("--no-session-persistence".to_string()),
    }
    args
}

fn initialize_request(launch: &Launch) -> Value {
    let mut request = json!({"subtype": "initialize", "hooks": {}, "skills": []});
    if !launch.tools.is_empty() {
        request["sdkMcpServers"] = json!([SERVER_NAME]);
    }
    match &launch.system {
        SystemPrompt::Append(text) => {
            request["appendSystemPrompt"] = json!(text);
            // Prompt changes must take effect when a process resumes the same session.
            request["systemPromptSnapshot"] = json!(false);
        }
        SystemPrompt::Replace(text) => request["systemPrompt"] = json!([text]),
    }
    request
}

impl ClaudeProcess {
    /// Start `claude` and finish the initialize handshake.
    ///
    /// # Errors
    ///
    /// The spawn failure, a handshake that does not finish in time, or a too-old Claude Code.
    pub(crate) async fn start(launch: &Launch) -> Result<Self, String> {
        let mut command = Command::new(&launch.executable);
        command
            .args(launch_args(launch))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (key, value) in ISOLATION_ENV {
            command.env(key, value);
        }
        for key in CLEARED_ENV {
            command.env_remove(key);
        }
        let mut child = command.spawn().map_err(|error| {
            format!(
                "Could not start Claude Code ({}): {error}",
                launch.executable.display()
            )
        })?;
        let stdin = Arc::new(Mutex::new(child.stdin.take()));
        let stdout = child.stdout.take().ok_or("Claude Code has no stdout")?;
        let stderr = child.stderr.take().ok_or("Claude Code has no stderr")?;

        let stderr_tail = Arc::new(std::sync::Mutex::new(String::new()));
        let tail = stderr_tail.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr);
            let mut buffer = [0u8; 4096];
            while let Ok(read) = reader.read(&mut buffer).await {
                if read == 0 {
                    break;
                }
                let mut tail = tail.lock().expect("stderr tail lock");
                tail.push_str(&String::from_utf8_lossy(&buffer[..read]));
                if tail.len() > STDERR_TAIL {
                    let cut = tail.len() - STDERR_TAIL;
                    let cut = (cut..tail.len())
                        .find(|index| tail.is_char_boundary(*index))
                        .unwrap_or(tail.len());
                    tail.drain(..cut);
                }
            }
        });

        let (sender, inbound) = mpsc::unbounded_channel();
        let controls: Controls = Arc::default();
        let ended = Arc::new(AtomicBool::new(false));
        tokio::spawn(read_stdout(
            stdout,
            ReaderState {
                stdin: stdin.clone(),
                sender,
                controls: controls.clone(),
                ended: ended.clone(),
                tools: mcp_tools(&launch.tools),
                stderr_tail,
            },
        ));

        let mut process = Self {
            child,
            stdin,
            inbound,
            controls,
            ended,
            next_request: AtomicU64::new(0),
            session_id: match &launch.persistence {
                Persistence::New(id) | Persistence::Resume(id) => Some(id.clone()),
                Persistence::Discard => None,
            },
            init: Value::Null,
        };
        let init =
            match tokio::time::timeout(START_TIMEOUT, process.request(initialize_request(launch)))
                .await
            {
                Ok(Ok(init)) => init,
                Ok(Err(error)) => {
                    process.kill().await;
                    return Err(error);
                }
                Err(_) => {
                    process.kill().await;
                    return Err("Claude Code did not start within 30 seconds".to_string());
                }
            };
        if let Some(error) = version_error(init.get("claude_code_version").and_then(Value::as_str))
        {
            process.kill().await;
            return Err(error);
        }
        process.init = init;
        Ok(process)
    }

    pub(crate) fn ended(&self) -> bool {
        self.ended.load(Ordering::SeqCst)
    }

    /// Whether the login is a Claude subscription (covered by the plan, so shown at $0).
    pub(crate) fn subscription(&self) -> bool {
        self.init
            .get("account")
            .and_then(|account| account.get("subscriptionType"))
            .is_some_and(|kind| !kind.is_null())
    }

    /// Send a control request and wait for its response.
    ///
    /// # Errors
    ///
    /// The control error Claude Code returned, or a closed process.
    pub(crate) async fn request(&self, request: Value) -> Result<Value, String> {
        let id = format!("prime-{}", self.next_request.fetch_add(1, Ordering::SeqCst));
        let (sender, receiver) = oneshot::channel();
        self.controls
            .lock()
            .expect("controls lock")
            .insert(id.clone(), sender);
        write_line(
            &self.stdin,
            &json!({"type": "control_request", "request_id": id, "request": request}),
        )
        .await?;
        receiver
            .await
            .map_err(|_| "Claude Code stopped before answering".to_string())?
    }

    /// Ask the running turn to stop without waiting for the answer (the `result` follows).
    pub(crate) async fn interrupt(&self) -> Result<(), String> {
        let id = format!("prime-{}", self.next_request.fetch_add(1, Ordering::SeqCst));
        write_line(
            &self.stdin,
            &json!({"type": "control_request", "request_id": id, "request": {"subtype": "interrupt"}}),
        )
        .await
    }

    /// Send user content. `next` queues it for the running turn (steering) instead of a new turn.
    pub(crate) async fn send_user(&self, content: Vec<Value>, next: bool) -> Result<(), String> {
        let mut message = json!({
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": null,
            "session_id": "",
            // Prime Agent already expanded its own commands; Claude Code must not read them again.
            "client_composed": true,
        });
        if next {
            message["priority"] = json!("next");
        }
        write_line(&self.stdin, &message).await
    }

    /// Answer a waiting `tools/call`.
    pub(crate) async fn answer_tool(
        &self,
        call: &ToolCallRequest,
        result: &Value,
    ) -> Result<(), String> {
        write_line(
            &self.stdin,
            &control_success(&call.control, &mcp_reply(&call.rpc, result)),
        )
        .await
    }

    async fn kill(&mut self) {
        // The process may already be gone; there is nothing left to stop then.
        let _ = self.child.kill().await;
    }

    /// Close input so Claude Code saves the conversation and exits; kill it if it lingers.
    pub(crate) async fn stop(&mut self) {
        self.stdin.lock().await.take();
        if tokio::time::timeout(STOP_TIMEOUT, self.child.wait())
            .await
            .is_err()
        {
            self.kill().await;
        }
    }
}

struct ReaderState {
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    sender: mpsc::UnboundedSender<Inbound>,
    controls: Controls,
    ended: Arc<AtomicBool>,
    tools: Value,
    stderr_tail: Arc<std::sync::Mutex<String>>,
}

/// The MCP handshake is answered here; `tools/call` goes to the session.
async fn handle_mcp(state: &ReaderState, request_id: &str, message: &Value) {
    let Some(id) = message.get("id") else {
        // A notification still gets a control response, or Claude Code waits for it.
        let reply = json!({"mcp_response": {"jsonrpc": "2.0", "result": {}, "id": 0}});
        let _ = write_line(&state.stdin, &control_success(request_id, &reply)).await;
        return;
    };
    let reply = match message.get("method").and_then(Value::as_str).unwrap_or("") {
        "initialize" => mcp_reply(
            id,
            &json!({
                "protocolVersion": message.pointer("/params/protocolVersion").cloned().unwrap_or(json!("2025-06-18")),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION")},
            }),
        ),
        "tools/list" => mcp_reply(id, &state.tools),
        "ping" => mcp_reply(id, &json!({})),
        "tools/call" => {
            let tool_use = message
                .pointer("/params/_meta/claudecode~1toolUseId")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let _ = state.sender.send(Inbound::ToolCall(ToolCallRequest {
                control: request_id.to_string(),
                rpc: id.clone(),
                tool_use,
            }));
            return;
        }
        method => {
            json!({"mcp_response": {"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": format!("Method not found: {method}")}}})
        }
    };
    // A write failure means the process is gone; the reader reports that when stdout closes.
    let _ = write_line(&state.stdin, &control_success(request_id, &reply)).await;
}

async fn read_stdout(stdout: tokio::process::ChildStdout, state: ReaderState) {
    let mut lines = BufReader::new(stdout).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            tracing::debug!(line, "claude-agent: ignoring a non-JSON line");
            continue;
        };
        match message.get("type").and_then(Value::as_str).unwrap_or("") {
            "control_response" => {
                let response = message.get("response").cloned().unwrap_or_default();
                let id = response
                    .get("request_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let sender = state.controls.lock().expect("controls lock").remove(id);
                if let Some(sender) = sender {
                    let result = if response.get("subtype").and_then(Value::as_str) == Some("error")
                    {
                        Err(response
                            .get("error")
                            .and_then(Value::as_str)
                            .unwrap_or("Claude Code control request failed")
                            .to_string())
                    } else {
                        Ok(response.get("response").cloned().unwrap_or_default())
                    };
                    let _ = sender.send(result);
                }
            }
            "control_request" => {
                let request_id = message
                    .get("request_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let request = message.get("request").cloned().unwrap_or_default();
                let is_ours = request.get("subtype").and_then(Value::as_str) == Some("mcp_message")
                    && request.get("server_name").and_then(Value::as_str) == Some(SERVER_NAME);
                if is_ours {
                    let inner = request.get("message").cloned().unwrap_or_default();
                    handle_mcp(&state, &request_id, &inner).await;
                } else {
                    let reply = json!({"type": "control_response", "response": {"subtype": "error", "request_id": request_id, "error": "Not supported by Prime Agent"}});
                    let _ = write_line(&state.stdin, &reply).await;
                }
            }
            "control_cancel_request" => {}
            _ => {
                let _ = state.sender.send(Inbound::Message(message));
            }
        }
    }
    state.ended.store(true, Ordering::SeqCst);
    for (_, sender) in state.controls.lock().expect("controls lock").drain() {
        let _ = sender.send(Err("Claude Code stopped before answering".to_string()));
    }
    let tail = state
        .stderr_tail
        .lock()
        .expect("stderr tail lock")
        .trim()
        .to_string();
    let _ = state.sender.send(Inbound::Exited(tail));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_below_the_minimum_are_refused() {
        assert_eq!(version_error(Some("2.1.316")), None);
        assert_eq!(version_error(Some("2.1.248-beta")), None);
        assert_eq!(version_error(Some("3.0.0")), None);
        assert!(version_error(Some("2.1.247")).is_some());
        assert!(version_error(None).is_some());
    }

    #[test]
    fn session_ids_are_version_four_uuids() {
        let id = new_session_id();
        assert_eq!(id.len(), 36);
        assert_eq!(&id[14..15], "4");
        assert!(matches!(&id[19..20], "8" | "9" | "a" | "b"));
    }
}
