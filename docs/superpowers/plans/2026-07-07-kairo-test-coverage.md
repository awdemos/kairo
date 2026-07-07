# Kairo Test Coverage Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add focused tests for currently untested public APIs and new architectural seams introduced in the deepening pass, without changing production behavior.

**Architecture:** Tests are unit/integration style using real objects and in-memory fakes. HTTP provider tests use `reqwest` mock infrastructure. Agent/orchestrator tests use fake `Provider`/`Tool`/`TaskRunner` implementations. API tests use `tower::ServiceExt` against the `axum` `Router`.

**Tech Stack:** Rust, tokio, axum, tower, reqwest, serde_json.

---

## Task 1: Config loading tests (kairo-core)

**Files:**
- Modify: `kairo-core/src/config.rs`

- [ ] **Step 1: Add failing test for `KairoConfig::from_env`**

Append to `kairo-core/src/config.rs` test module:

```rust
#[test]
fn test_provider_config_from_env() {
    // Given
    temp_env::with_vars(
        [("OPENAI_API_KEY", Some("sk-openai")), ("ANTHROPIC_API_KEY", Some("sk-anthropic"))],
        || {
            // When
            let config = ProviderConfig::from_env();
            // Then
            assert_eq!(config.openai_api_key, "sk-openai");
            assert_eq!(config.anthropic_api_key, "sk-anthropic");
        },
    );
}

#[test]
fn test_api_config_from_env() {
    temp_env::with_vars([("KAIRO_API_PORT", Some("8080"))], || {
        let config = ApiConfig::from_env();
        assert_eq!(config.port, 8080);
    });
}

#[test]
fn test_kairo_config_load_uses_defaults() {
    let config = KairoConfig::load();
    assert_eq!(config.api.port, 3000);
    assert_eq!(config.agent.max_tokens, 4096);
}
```

- [ ] **Step 2: Add `temp-env` dev dependency to `kairo-core/Cargo.toml`**

```toml
[dev-dependencies]
temp-env = "0.3"
```

- [ ] **Step 3: Run tests and confirm failure**

Run: `cargo test -p kairo-core`
Expected: FAIL because `temp-env` not yet available or `config` test module does not exist. If no `#[cfg(test)]` module exists, create one at the bottom of `config.rs`.

- [ ] **Step 4: Implement minimal passing test module**

Add:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // ... tests from Step 1
}
```

- [ ] **Step 5: Run tests and confirm pass**

Run: `cargo test -p kairo-core`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add kairo-core/src/config.rs kairo-core/Cargo.toml
git commit -m "test(core): add config loading tests"
```

---

## Task 2: Provider response parsing tests (kairo-providers)

**Files:**
- Modify: `kairo-providers/src/openai.rs`, `kairo-providers/src/anthropic.rs`, `kairo-providers/src/gemini.rs`
- Modify: `kairo-providers/Cargo.toml`

- [ ] **Step 1: Add `mockito` dev dependency**

```toml
[dev-dependencies]
mockito = "1.6"
tokio-test = { workspace = true }
```

- [ ] **Step 2: Add failing OpenAI response test**

In `kairo-providers/src/openai.rs` test module:

```rust
#[tokio::test]
async fn test_openai_provider_parses_successful_response() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"choices":[{"message":{"content":"hello"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#)
        .create_async()
        .await;

    let provider = OpenAiProvider::new("test-key", "gpt-4o").with_base_url(server.url());
    let response = provider
        .complete(vec![Message { role: Role::User, content: "hi".into(), name: None, timestamp: Utc::now() }], CompletionOptions::default())
        .await
        .unwrap();

    assert_eq!(response.content, "hello");
    assert_eq!(response.usage.total_tokens, 7);
}
```

- [ ] **Step 3: Run test and confirm failure**

Run: `cargo test -p kairo-providers openai_provider_parses`
Expected: FAIL because `with_base_url` may need to adjust endpoint path or `Message` import missing.

- [ ] **Step 4: Fix imports and make test pass**

Ensure test module imports `kairo_core::{Message, Role, CompletionOptions}` and `chrono::Utc`.

- [ ] **Step 5: Add analogous Anthropic and Gemini tests**

In `anthropic.rs` and `gemini.rs`, add tests using `mockito` with provider-specific response JSON.

- [ ] **Step 6: Run all provider tests**

Run: `cargo test -p kairo-providers`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add kairo-providers/src/openai.rs kairo-providers/src/anthropic.rs kairo-providers/src/gemini.rs kairo-providers/Cargo.toml
git commit -m "test(providers): add mock HTTP response parsing tests"
```

---

## Task 3: Agent runtime tests with fakes (kairo-agents)

**Files:**
- Modify: `kairo-agents/src/react_loop.rs`, `kairo-agents/src/tool_dispatcher.rs`

- [ ] **Step 1: Add failing ReActLoop test**

In `kairo-agents/src/react_loop.rs` test module, create a fake `Provider` and `Tool`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use kairo_core::{CompletionOptions, CompletionResponse, Message, Provider, Role, Tool, ToolInput, ToolOutput};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct FakeProvider {
        responses: Vec<String>,
        calls: AtomicUsize,
    }

    impl Provider for FakeProvider {
        fn complete<'a>(
            &'a self,
            _messages: Vec<Message>,
            _options: CompletionOptions,
        ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, KairoError>> + Send + 'a>> {
            let idx = self.calls.fetch_add(1, Ordering::SeqCst);
            let content = self.responses.get(idx).cloned().unwrap_or_default();
            Box::pin(async move {
                Ok(CompletionResponse {
                    content,
                    usage: TokenUsage::default(),
                    model: ModelId::Gpt4o,
                    finish_reason: "stop".into(),
                })
            })
        }
    }

    struct FakeTool;
    impl Tool for FakeTool {
        fn name(&self) -> &str { "fake_tool" }
        fn description(&self) -> &str { "fake" }
        fn execute<'a>(&'a self, _input: ToolInput) -> Pin<Box<dyn Future<Output = Result<ToolOutput, KairoError>> + Send + 'a>> {
            Box::pin(async move {
                Ok(ToolOutput { success: true, result: serde_json::json!({"value": 42}) })
            })
        }
    }

    #[tokio::test]
    async fn test_react_loop_finishes_on_action_finish() {
        let provider = Arc::new(FakeProvider { responses: vec!["Thought: done\nAction: finish".into()], calls: AtomicUsize::new(0) });
        let council = ModelCouncil::new();
        council.register_adapter(ModelId::Gpt4o, provider.clone()).await;
        let tools = Arc::new(ToolRegistry::new());
        let memory = Arc::new(HybridMemory::new());
        let agent = Agent { id: Uuid::new_v4(), model: ModelId::Gpt4o, config: AgentConfig::default() };
        let loop_ = ReActLoop::new(agent, memory, tools, Arc::new(council), 10);
        let result = loop_.run(Context::new(Uuid::new_v4())).await.unwrap();
        assert_eq!(result.output, "done");
    }
}
```

Note: `ModelCouncil::register_adapter` may not exist. If not, use `ModelCouncil::new()` with `with_api_key` and a custom `Provider` by calling `complete` directly? The `ReActLoop` takes a `ModelCouncil`, so we need a way to inject a fake. If `ModelCouncil` does not support injecting providers, add a `register_provider(model, provider)` method to it in `kairo-council/src/lib.rs` as part of this task (it is a testability improvement).

- [ ] **Step 2: Add `register_provider` test helper to ModelCouncil if needed**

In `kairo-council/src/lib.rs`:

```rust
pub async fn register_provider(&self, model: ModelId, provider: Arc<dyn Provider>) {
    let mut providers = self.providers.write().await;
    providers.insert(model, provider);
}
```

- [ ] **Step 3: Run test and confirm failure**

Run: `cargo test -p kairo-agents react_loop_finishes`
Expected: FAIL until imports/helpers exist.

- [ ] **Step 4: Implement helpers and make test pass**

- [ ] **Step 5: Add ToolDispatcher test**

In `kairo-agents/src/tool_dispatcher.rs`:

```rust
#[tokio::test]
async fn test_tool_dispatcher_routes_to_registered_tool() {
    let registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool)).await;
    let dispatcher = ToolDispatcher::new(registry);
    let call = ToolCall { tool_name: "calculator".into(), arguments: serde_json::json!({"expression": "1+1"}) };
    let output = dispatcher.dispatch(&call).await.unwrap();
    assert!(output.success);
}
```

- [ ] **Step 6: Run all agent tests**

Run: `cargo test -p kairo-agents`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add kairo-agents/src/react_loop.rs kairo-agents/src/tool_dispatcher.rs kairo-council/src/lib.rs
git commit -m "test(agents): add ReActLoop and ToolDispatcher tests with fakes"
```

---

## Task 4: Orchestrator executor end-to-end with fake runner (kairo-orchestrator)

**Files:**
- Modify: `kairo-orchestrator/src/executor.rs`, `kairo-orchestrator/src/lib.rs`

- [ ] **Step 1: Add failing executor test**

In `kairo-orchestrator/src/executor.rs` test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use kairo_core::{AgentResult, Context, KairoError, Output, Task, Workflow, WorkflowStatus};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use uuid::Uuid;

    struct FakeRunner {
        calls: AtomicUsize,
    }
    #[async_trait::async_trait]
    impl TaskRunner for FakeRunner {
        async fn run(&self, _ctx: Context) -> Result<AgentResult, KairoError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(AgentResult {
                output: "done".into(),
                thoughts: vec![],
                tool_calls: vec![],
                token_usage: TokenUsage::default(),
            })
        }
    }

    #[tokio::test]
    async fn test_executor_runs_single_task() {
        let workflow = Workflow {
            id: Uuid::new_v4(),
            name: "test".into(),
            tasks: vec![Task { id: Uuid::new_v4(), description: "task 1".into(), status: TaskStatus::Pending }],
            subtasks: vec![],
            status: WorkflowStatus::Draft,
        };
        let store = InMemoryWorkflowStore::new();
        let runner = Arc::new(FakeRunner { calls: AtomicUsize::new(0) });
        let executor = Executor::new(store.clone(), runner.clone());
        let result = executor.execute(workflow.id, workflow, Context::new(Uuid::new_v4())).await.unwrap();
        assert_eq!(result.status, WorkflowStatus::Completed);
        assert_eq!(runner.calls.load(Ordering::SeqCst), 1);
    }
}
```

- [ ] **Step 2: Run test and confirm failure**

Run: `cargo test -p kairo-orchestrator executor_runs_single_task`
Expected: FAIL until imports and `Executor::execute` signature support workflow + id.

- [ ] **Step 3: Adjust executor or test to match actual API**

If `Executor::execute` takes only `workflow_id: Uuid` and `ctx: Context`, ensure the store already holds the workflow (register first). Update test accordingly.

- [ ] **Step 4: Run all orchestrator tests**

Run: `cargo test -p kairo-orchestrator`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kairo-orchestrator/src/executor.rs
git commit -m "test(orchestrator): add executor end-to-end test with fake runner"
```

---

## Task 5: Filesystem and code execution tool tests (kairo-tools)

**Files:**
- Modify: `kairo-tools/src/lib.rs`

- [ ] **Step 1: Add failing filesystem tool test**

Append to existing test module:

```rust
#[tokio::test]
async fn test_filesystem_tool_reads_file() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("kairo-test-{}", uuid::Uuid::new_v4()));
    std::fs::write(&path, "hello file").unwrap();

    let tool = FileSystemTool;
    let input = ToolInput {
        arguments: serde_json::json!({"operation": "read", "path": path.to_str().unwrap()}),
    };
    let output = tool.execute(input).await.unwrap();
    assert!(output.success);
    assert_eq!(output.result.get("content").unwrap().as_str().unwrap(), "hello file");

    std::fs::remove_file(&path).unwrap();
}
```

- [ ] **Step 2: Add failing code execution tool test**

```rust
#[tokio::test]
async fn test_code_execution_tool_runs_python() {
    let tool = CodeExecutionTool;
    let input = ToolInput {
        arguments: serde_json::json!({"language": "python", "code": "print(2+2)"}),
    };
    let output = tool.execute(input).await.unwrap();
    assert!(output.success);
    let stdout = output.result.get("stdout").unwrap().as_str().unwrap();
    assert!(stdout.contains('4'));
}
```

- [ ] **Step 3: Run tests and confirm failures**

Run: `cargo test -p kairo-tools`
Expected: FAIL if tool schemas differ; inspect actual JSON keys.

- [ ] **Step 4: Adjust tests to actual tool schemas**

Read `FileSystemTool::execute` and `CodeExecutionTool::execute` to determine correct JSON arguments and result shape.

- [ ] **Step 5: Run all tool tests**

Run: `cargo test -p kairo-tools`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add kairo-tools/src/lib.rs
git commit -m "test(tools): add filesystem and code execution tool tests"
```

---

## Task 6: HTTP API handler tests (kairo-api)

**Files:**
- Modify: `kairo-api/src/lib.rs`
- Modify: `kairo-api/Cargo.toml`

- [ ] **Step 1: Add tower-http dev dependency if needed**

`axum` already re-exports `tower::ServiceExt`. No new dep needed unless `http_body_util` required.

- [ ] **Step 2: Add failing health and chat handler tests**

Append to existing test module in `kairo-api/src/lib.rs`:

```rust
#[tokio::test]
async fn test_health_check() {
    let state = ApiState {
        agents: Arc::new(RwLock::new(vec![])),
        engine: Arc::new(WorkflowEngine::new()),
        telemetry: kairo_telemetry::build_default(),
    };
    let app = app(state);
    let response = app
        .oneshot(axum::http::Request::builder().uri("/health").body(axum::body::Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
```

- [ ] **Step 3: Run test and confirm failure**

Run: `cargo test -p kairo-api test_health_check`
Expected: FAIL due to missing imports or body handling.

- [ ] **Step 4: Fix imports and make test pass**

Add `use axum::body::Body;` and `use tower::ServiceExt;` in test module.

- [ ] **Step 5: Add chat handler error test**

```rust
#[tokio::test]
async fn test_chat_handler_returns_503_when_no_agents() {
    let state = ApiState {
        agents: Arc::new(RwLock::new(vec![])),
        engine: Arc::new(WorkflowEngine::new()),
        telemetry: kairo_telemetry::build_default(),
    };
    let app = app(state);
    let body = serde_json::to_vec(&ChatRequest { messages: vec![], model: None, stream: None }).unwrap();
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}
```

- [ ] **Step 6: Run all API tests**

Run: `cargo test -p kairo-api`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add kairo-api/src/lib.rs kairo-api/Cargo.toml
git commit -m "test(api): add HTTP handler tests for health and chat endpoints"
```

---

## Task 7: Final verification

- [ ] **Step 1: Run full workspace test suite**

Run: `cargo test --workspace`
Expected: all tests pass

- [ ] **Step 2: Run workspace check**

Run: `cargo check --workspace --all-targets --all-features`
Expected: clean

- [ ] **Step 3: Run clippy with empty config**

Run: `CLIPPY_CONF_DIR=$(mktemp -d) cargo clippy --workspace --all-targets --all-features -- -D warnings`
Expected: clean

- [ ] **Step 4: Commit if all pass**

If no additional changes needed, no extra commit. Otherwise commit fixes.
