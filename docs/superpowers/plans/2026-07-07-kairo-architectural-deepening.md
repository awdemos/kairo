# Kairo Architectural Deepening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen six shallow architectural seams in the Kairo workspace so provider selection, model identification, agent reasoning, workflow execution, error classification, and telemetry setup stop being bug-hiding pockets.

**Architecture:** Each task introduces a small, testable interface (seam) and moves the messy implementation behind an adapter. Public APIs stay source-compatible where possible; behavior changes only where the current code is actively misleading (e.g., every error mapping to HTTP 500).

**Tech Stack:** Rust 2021, tokio, axum, thiserror, tracing, serde, petgraph.

**Execution order:** The tasks are intentionally ordered to minimize re-work. Task 1 (structured errors) is foundational. Task 2 (model resolver) feeds Task 3 (provider registry). Tasks 4 and 5 are largely independent once errors are clean. Task 6 touches CLI/API wiring and should come after API error mapping in Task 1.

---

## Task 1: Structured `KairoError` variants and provider identity

**Files:**
- Create: `kairo-core/src/provider_error.rs`
- Create: `kairo-core/src/error_renderer.rs`
- Modify: `kairo-core/src/error.rs`
- Modify: `kairo-core/src/lib.rs`
- Modify: `kairo-providers/src/lib.rs`
- Modify: `kairo-api/src/lib.rs`

**Context:**
Current `KairoError` wraps plain `String` messages. Provider errors lose HTTP status, provider identity, and retryability. The API handler maps every provider error to `500 INTERNAL_SERVER_ERROR`.

**Steps:**

- [ ] **Step 1.1: Write the failing test for structured provider errors**

Create `kairo-core/src/provider_error.rs` with a test first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_records_identity_and_retryability() {
        let err = ProviderError::new("openai", "gpt-4o")
            .with_status(429)
            .with_retryable(true)
            .with_message("rate limited");
        assert_eq!(err.identity.provider, "openai");
        assert_eq!(err.identity.model, "gpt-4o");
        assert_eq!(err.status, Some(429));
        assert!(err.retryable);
        assert_eq!(err.message, "rate limited");
    }
}
```

- [ ] **Step 1.2: Run the test and confirm it fails**

Run: `cargo test -p kairo-core provider_error_records_identity`
Expected: FAIL (module/type not found).

- [ ] **Step 1.3: Implement `ProviderError` and `ProviderIdentity`**

In `kairo-core/src/provider_error.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderIdentity {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderError {
    pub identity: ProviderIdentity,
    pub status: Option<u16>,
    pub retryable: bool,
    pub message: String,
}

impl ProviderError {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            identity: ProviderIdentity {
                provider: provider.into(),
                model: model.into(),
            },
            status: None,
            retryable: false,
            message: String::new(),
        }
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }
}
```

- [ ] **Step 1.4: Update `KairoError` to carry structured payloads**

In `kairo-core/src/error.rs`, replace the string variants for provider/tool/connector with typed payloads:

```rust
use crate::provider_error::{ProviderError};

#[derive(Error, Debug, Clone, PartialEq)]
pub enum KairoError {
    #[error("Provider error: {0}")]
    Provider(ProviderError),

    #[error("Tool error: {name}: {message}")]
    Tool { name: String, message: String },

    #[error("Connector error: {name}: {message}")]
    Connector { name: String, message: String },

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Routing error: {0}")]
    Routing(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Workflow error: {0}")]
    Workflow(String),

    #[error("Agent error: {0}")]
    Agent(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Timeout after {0}s")]
    Timeout(u64),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Unauthorized")]
    Unauthorized,

    #[error("Rate limited")]
    RateLimited,

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Sandbox error: {0}")]
    Sandbox(String),
}
```

Keep `KairoError::Unauthorized` and `KairoError::RateLimited` for top-level routing convenience.

- [ ] **Step 1.5: Re-export new types in `kairo-core/src/lib.rs`**

Add:

```rust
pub mod provider_error;
pub use provider_error::{ProviderError, ProviderIdentity};
```

- [ ] **Step 1.6: Update provider adapters to produce `ProviderError`**

In `kairo-providers/src/lib.rs`, replace `KairoError::Provider(format!(...))` with structured errors. Example for OpenAI:

```rust
KairoError::Provider(
    ProviderError::new("openai", &self.model)
        .with_status(status.as_u16())
        .with_retryable(is_retryable(status.as_u16()))
        .with_message(text),
)
```

Introduce a small helper `is_retryable(status: u16) -> bool` returning true for 429/500/502/503.

- [ ] **Step 1.7: Update API error mapping**

In `kairo-api/src/lib.rs`, replace the blanket `INTERNAL_SERVER_ERROR` with a renderer:

```rust
fn into_response(err: KairoError) -> (StatusCode, Json<ErrorResponse>) {
    let status = match &err {
        KairoError::Provider(p) if p.status == Some(401) => StatusCode::UNAUTHORIZED,
        KairoError::Provider(p) if p.status == Some(429) => StatusCode::TOO_MANY_REQUESTS,
        KairoError::Provider(p) if p.retryable => StatusCode::SERVICE_UNAVAILABLE,
        KairoError::RateLimited => StatusCode::TOO_MANY_REQUESTS,
        KairoError::Unauthorized => StatusCode::UNAUTHORIZED,
        KairoError::NotFound(_) => StatusCode::NOT_FOUND,
        KairoError::Validation(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    error!(error = %err, "request failed");
    (status, Json(ErrorResponse { error: err.to_string() }))
}
```

Then use `.map_err(into_response)?` in the chat handler.

- [ ] **Step 1.8: Fix all downstream pattern matches**

Run: `cargo check --workspace --all-targets --all-features`
Fix any compile errors from the changed variants (e.g., `KairoError::Provider(s)` -> `KairoError::Provider(p)` and adjust usage).

- [ ] **Step 1.9: Add error renderer tests**

In `kairo-api/src/lib.rs` (inside `#[cfg(test)] mod tests`):

```rust
#[test]
fn provider_429_maps_to_too_many_requests() {
    let err = KairoError::Provider(
        ProviderError::new("openai", "gpt-4o")
            .with_status(429)
            .with_retryable(true)
            .with_message("rate limited"),
    );
    let (status, _) = into_response(err);
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
}
```

- [ ] **Step 1.10: Run tests and commit**

Run: `cargo test -p kairo-core -p kairo-providers -p kairo-api`
Expected: PASS.
Commit:

```bash
git add kairo-core/src kairo-providers/src/lib.rs kairo-api/src/lib.rs
git commit -m "feat(core): structured provider errors with identity and status mapping"
```

---

## Task 2: Central `ModelId` resolver

**Files:**
- Create: `kairo-core/src/model_id.rs`
- Modify: `kairo-core/src/lib.rs`
- Modify: `kairo-core/src/model.rs`
- Modify: `kairo-cli/src/lib.rs`
- Modify: `kairo-providers/src/lib.rs`
- Modify: `kairo-council/src/lib.rs`

**Context:**
String-to-`ModelId` mapping is duplicated in CLI, provider factory, and council defaults. Unknown strings silently become `Custom`, which can route to the wrong provider later.

**Steps:**

- [ ] **Step 2.1: Write the failing test for model resolution**

Create `kairo-core/src/model_id.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_known_model_strings() {
        assert_eq!(ModelId::resolve("gpt-4o"), Some(ModelId::Gpt4o));
        assert_eq!(ModelId::resolve("claude-3-5-sonnet"), Some(ModelId::Claude3_5Sonnet));
        assert_eq!(ModelId::resolve("gemini-2.0-flash"), Some(ModelId::Gemini2_0Flash));
    }

    #[test]
    fn custom_models_are_resolved_by_provider_prefix() {
        assert_eq!(ModelId::resolve("gpt-fake"), Some(ModelId::Custom("gpt-fake".into())));
        assert_eq!(ModelId::resolve("claude-fake"), Some(ModelId::Custom("claude-fake".into())));
        assert_eq!(ModelId::resolve("gemini-fake"), Some(ModelId::Custom("gemini-fake".into())));
    }

    #[test]
    fn unknown_prefix_returns_none() {
        assert_eq!(ModelId::resolve("totally-unknown"), None);
    }
}
```

- [ ] **Step 2.2: Run the test and confirm it fails**

Run: `cargo test -p kairo-core resolves_known_model_strings`
Expected: FAIL (method not found).

- [ ] **Step 2.3: Implement `ModelId::resolve`**

In `kairo-core/src/model_id.rs`:

```rust
use crate::model::ModelId;

impl ModelId {
    pub fn resolve(name: &str) -> Option<Self> {
        match name {
            "gpt-4o" => Some(ModelId::Gpt4o),
            "gpt-4o-mini" => Some(ModelId::Gpt4oMini),
            "gpt-4" => Some(ModelId::Gpt4),
            "gpt-4-turbo" => Some(ModelId::Gpt4Turbo),
            "gpt-3.5-turbo" => Some(ModelId::Gpt3_5Turbo),
            "o1" => Some(ModelId::O1),
            "o1-mini" => Some(ModelId::O1Mini),
            "o3" => Some(ModelId::O3),
            "o3-mini" => Some(ModelId::O3Mini),
            "o4" => Some(ModelId::O4),
            "o4-mini" => Some(ModelId::O4Mini),
            "claude-3-5-sonnet" => Some(ModelId::Claude3_5Sonnet),
            "claude-3-opus" => Some(ModelId::Claude3Opus),
            "claude-3-haiku" => Some(ModelId::Claude3Haiku),
            "claude-3-5-haiku" => Some(ModelId::Claude3_5Haiku),
            "claude-4" => Some(ModelId::Claude4),
            "claude-4-opus" => Some(ModelId::Claude4Opus),
            "gemini-2.0-flash" => Some(ModelId::Gemini2_0Flash),
            "gemini-2.0-pro" => Some(ModelId::Gemini2_0Pro),
            "gemini-2.5-flash" => Some(ModelId::Gemini2_5Flash),
            "gemini-2.5-pro" => Some(ModelId::Gemini2_5Pro),
            "gemini-1.5-pro" => Some(ModelId::Gemini1_5Pro),
            "gemini-1.5-flash" => Some(ModelId::Gemini1_5Flash),
            _ => {
                if name.starts_with("gpt-") || name.starts_with("o1") || name.starts_with("o3") || name.starts_with("o4") {
                    Some(ModelId::Custom(name.to_string()))
                } else if name.starts_with("claude-") {
                    Some(ModelId::Custom(name.to_string()))
                } else if name.starts_with("gemini-") {
                    Some(ModelId::Custom(name.to_string()))
                } else {
                    None
                }
            }
        }
    }
}
```

- [ ] **Step 2.4: Re-export in `kairo-core/src/lib.rs`**

Add:

```rust
pub mod model_id;
```

- [ ] **Step 2.5: Replace `parse_model` in CLI**

In `kairo-cli/src/lib.rs`:

```rust
fn parse_model(name: &str) -> Result<ModelId, KairoError> {
    ModelId::resolve(name)
        .ok_or_else(|| KairoError::Model(format!("Unknown model: {}", name)))
}
```

Update all callers to handle the `Result` (e.g., propagate or default to a known model).

- [ ] **Step 2.6: Replace provider factory selection with `ModelId::resolve` fallback**

In `kairo-providers/src/lib.rs`, for `ModelId::Custom(name)` handling, call `ModelId::resolve(name)` first to classify known aliases, then route by prefix. This removes duplicated prefix logic.

- [ ] **Step 2.7: Update council defaults to use resolver**

In `kairo-council/src/lib.rs`, ensure `default_council` uses `ModelId` variants directly; no string parsing here. If council needs string-based registration, use `ModelId::resolve`.

- [ ] **Step 2.8: Run workspace check and tests**

Run: `cargo test -p kairo-core -p kairo-cli -p kairo-providers -p kairo-council`
Expected: PASS.

- [ ] **Step 2.9: Commit**

```bash
git add kairo-core/src kairo-cli/src/lib.rs kairo-providers/src/lib.rs kairo-council/src/lib.rs
git commit -m "feat(core): central ModelId resolver eliminates duplicate string parsing"
```

---

## Task 3: Provider Registry seam

**Files:**
- Create: `kairo-providers/src/registry.rs`
- Modify: `kairo-providers/src/lib.rs`
- Modify: `kairo-providers/Cargo.toml` (if new deps needed — none expected)

**Context:**
`create_provider` is a large match block that knows every provider family. Adding a provider requires editing the central factory.

**Steps:**

- [ ] **Step 3.1: Write the failing test for the registry**

Create `kairo-providers/src/registry.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use kairo_core::model::ModelId;

    #[test]
    fn registry_resolves_openai_models() {
        let registry = ProviderRegistry::default();
        let provider = registry.resolve(&ModelId::Gpt4o, "key");
        assert!(provider.is_ok());
    }

    #[test]
    fn registry_rejects_unknown_models() {
        let registry = ProviderRegistry::default();
        let provider = registry.resolve(&ModelId::Custom("totally-unknown".into()), "key");
        assert!(provider.is_err());
    }
}
```

- [ ] **Step 3.2: Run the test and confirm it fails**

Run: `cargo test -p kairo-providers registry_resolves_openai_models`
Expected: FAIL.

- [ ] **Step 3.3: Implement the registry**

In `kairo-providers/src/registry.rs`:

```rust
use std::sync::Arc;
use kairo_core::error::KairoError;
use kairo_core::model::ModelId;
use kairo_core::traits::Provider;

use crate::{AnthropicProvider, GeminiProvider, OpenAiProvider};

pub struct ProviderRegistry {
    // Ordered by specificity: more specific matchers first.
    adapters: Vec<Box<dyn Fn(&ModelId) -> Option<Arc<dyn Provider>> + Send + Sync>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self { adapters: Vec::new() }
    }

    pub fn register<F>(mut self, matcher: F) -> Self
    where
        F: Fn(&ModelId) -> Option<Arc<dyn Provider>> + Send + Sync + 'static,
    {
        self.adapters.push(Box::new(matcher));
        self
    }

    pub fn resolve(&self, model: &ModelId, api_key: &str) -> Result<Arc<dyn Provider>, KairoError> {
        for adapter in &self.adapters {
            if let Some(provider) = adapter(model) {
                return Ok(provider);
            }
        }
        Err(KairoError::Model(format!(
            "No provider available for model: {:?}",
            model
        )))
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
            .register(|model| match model {
                ModelId::Gpt4o | ModelId::Gpt4oMini | ModelId::Gpt4 | ModelId::Gpt4Turbo | ModelId::Gpt3_5Turbo | ModelId::O1 | ModelId::O3Mini => {
                    // Need api_key; we can't capture it here. Instead store matchers as model-only and apply api_key in resolve.
                    None
                }
                _ => None,
            })
    }
}
```

The matcher-as-closure cannot capture `api_key`. Redesign: store struct adapters with a `matches(&ModelId) -> bool` and `build(&str) -> Arc<dyn Provider>`.

Correct implementation:

```rust
pub trait ProviderAdapter: Send + Sync {
    fn matches(&self, model: &ModelId) -> bool;
    fn build(&self, model: &ModelId, api_key: &str) -> Arc<dyn Provider>;
}

pub struct OpenAiAdapter;
impl ProviderAdapter for OpenAiAdapter {
    fn matches(&self, model: &ModelId) -> bool {
        matches!(model, ModelId::Gpt4o | ModelId::Gpt4oMini | ... | ModelId::Custom(name) if name.starts_with("gpt-") || ...)
    }
    fn build(&self, model: &ModelId, api_key: &str) -> Arc<dyn Provider> {
        Arc::new(OpenAiProvider::new(api_key, model.to_string()))
    }
}
// Similarly AnthropicAdapter, GeminiAdapter

pub struct ProviderRegistry {
    adapters: Vec<Box<dyn ProviderAdapter>>,
}

impl ProviderRegistry {
    pub fn with_defaults() -> Self {
        Self {
            adapters: vec![
                Box::new(OpenAiAdapter),
                Box::new(AnthropicAdapter),
                Box::new(GeminiAdapter),
            ],
        }
    }

    pub fn resolve(&self, model: &ModelId, api_key: &str) -> Result<Arc<dyn Provider>, KairoError> {
        for adapter in &self.adapters {
            if adapter.matches(model) {
                return Ok(adapter.build(model, api_key));
            }
        }
        Err(KairoError::Model(format!("No provider available for model: {:?}", model)))
    }
}
```

- [ ] **Step 3.4: Refactor `create_provider` to use the registry**

In `kairo-providers/src/lib.rs`:

```rust
pub fn create_provider(model: &ModelId, api_key: &str) -> Result<Arc<dyn Provider>, KairoError> {
    ProviderRegistry::with_defaults().resolve(model, api_key)
}
```

- [ ] **Step 3.5: Split provider implementations if `lib.rs` is oversized**

If `kairo-providers/src/lib.rs` exceeds 250 pure LOC, split into:
- `kairo-providers/src/openai.rs`
- `kairo-providers/src/anthropic.rs`
- `kairo-providers/src/gemini.rs`
- `kairo-providers/src/registry.rs`
- `kairo-providers/src/lib.rs` becomes a thin public facade.

This is recommended but optional; measure with `awk '!/^[[:space:]]*$/ && !/^[[:space:]]*\/\//' kairo-providers/src/lib.rs | wc -l`. If >250, split.

- [ ] **Step 3.6: Run tests and commit**

Run: `cargo test -p kairo-providers`
Expected: PASS.
Commit:

```bash
git add kairo-providers/src
git commit -m "feat(providers): introduce ProviderRegistry seam and adapter matcher"
```

---

## Task 4: Split `ReActAgent` into loop, parser, and dispatcher

**Files:**
- Create: `kairo-agents/src/thought_parser.rs`
- Create: `kairo-agents/src/tool_dispatcher.rs`
- Create: `kairo-agents/src/react_loop.rs`
- Modify: `kairo-agents/src/lib.rs`

**Context:**
`ReActAgent::run_inner` does council calls, parsing, tool execution, memory management, and message history in one function. `parse_thought` only reads the first line and ignores the structured format.

**Steps:**

- [ ] **Step 4.1: Write failing test for `ThoughtParser`**

Create `kairo-agents/src/thought_parser.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_thought_action_tool_and_observation() {
        let text = "Thought: I need to search\nAction: web_search\nTool: web_search(query=rust)\nObservation: results";
        let thought = ThoughtParser::default().parse(text).unwrap();
        assert_eq!(thought.thought, "I need to search");
        assert_eq!(thought.action, Some("web_search".to_string()));
        assert_eq!(thought.tool_calls.len(), 1);
        assert_eq!(thought.tool_calls[0].tool_name, "web_search");
    }

    #[test]
    fn finish_action_has_no_tools() {
        let text = "Thought: done\nAction: finish";
        let thought = ThoughtParser::default().parse(text).unwrap();
        assert_eq!(thought.action, Some("finish".to_string()));
        assert!(thought.tool_calls.is_empty());
    }
}
```

- [ ] **Step 4.2: Run and confirm failure**

Run: `cargo test -p kairo-agents parses_thought_action_tool`
Expected: FAIL.

- [ ] **Step 4.4: Implement `ThoughtParser` and `ToolDispatcher`**

In `kairo-agents/src/thought_parser.rs`:

```rust
use kairo_core::error::KairoError;
use kairo_core::types::ToolCall;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParsedThought {
    pub thought: String,
    pub action: Option<String>,
    pub observation: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Default)]
pub struct ThoughtParser;

impl ThoughtParser {
    pub fn parse(&self, content: &str) -> Result<ParsedThought, KairoError> {
        let mut thought = String::new();
        let mut action: Option<String> = None;
        let mut observation: Option<String> = None;
        let mut tool_calls = Vec::new();

        let mut current_field: Option<&str> = None;
        let mut buffer = String::new();

        for line in content.lines() {
            if let Some((key, value)) = line.split_once(':') {
                if let Some(field) = current_field {
                    Self::flush(field, &buffer, &mut thought, &mut action, &mut observation, &mut tool_calls)?;
                }
                current_field = Some(key.trim());
                buffer = value.trim().to_string();
            } else if let Some(field) = current_field {
                buffer.push('\n');
                buffer.push_str(line);
            }
        }
        if let Some(field) = current_field {
            Self::flush(field, &buffer, &mut thought, &mut action, &mut observation, &mut tool_calls)?;
        }

        Ok(ParsedThought { thought, action, observation, tool_calls })
    }

    fn flush(
        field: &str,
        value: &str,
        thought: &mut String,
        action: &mut Option<String>,
        observation: &mut Option<String>,
        tool_calls: &mut Vec<ToolCall>,
    ) -> Result<(), KairoError> {
        match field {
            "Thought" => *thought = value.to_string(),
            "Action" => *action = Some(value.to_string()),
            "Observation" => *observation = Some(value.to_string()),
            "Tool" => {
                let call = Self::parse_tool_call(value)?;
                tool_calls.push(call);
            }
            _ => {}
        }
        Ok(())
    }

    fn parse_tool_call(value: &str) -> Result<ToolCall, KairoError> {
        let (name, args) = value.split_once('(').ok_or_else(|| KairoError::Agent("Invalid tool call format".into()))?;
        let args = args.strip_suffix(')').ok_or_else(|| KairoError::Agent("Missing closing parenthesis in tool call".into()))?;
        let mut arguments = std::collections::HashMap::new();
        for part in args.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let (k, v) = part.split_once('=').ok_or_else(|| KairoError::Agent(format!("Invalid tool argument: {}", part)))?;
            arguments.insert(k.trim().to_string(), serde_json::Value::String(v.trim().to_string()));
        }
        Ok(ToolCall {
            tool_name: name.trim().to_string(),
            arguments,
        })
    }
}
```

In `kairo-agents/src/tool_dispatcher.rs`:

```rust
use kairo_core::error::KairoError;
use kairo_core::types::{ToolCall, ToolInput, ToolOutput};
use kairo_tools::ToolRegistry;

pub struct ToolDispatcher {
    registry: ToolRegistry,
}

impl ToolDispatcher {
    pub fn new(registry: ToolRegistry) -> Self {
        Self { registry }
    }

    pub async fn dispatch(&self, call: &ToolCall) -> Result<ToolOutput, KairoError> {
        let input = ToolInput {
            arguments: call.arguments.clone(),
        };
        self.registry.execute(&call.tool_name, input).await
    }
}
```

- [ ] **Step 4.5: Extract `ReActLoop`**

In `kairo-agents/src/react_loop.rs`:

```rust
use kairo_core::context::Context;
use kairo_core::error::KairoError;
use kairo_core::types::{AgentResult, CompletionOptions, Message, Role, Thought, TokenUsage};
use kairo_council::ModelCouncil;
use kairo_memory::Memory;
use tracing::{debug, error, info};

use crate::thought_parser::{ParsedThought, ThoughtParser};
use crate::tool_dispatcher::ToolDispatcher;

pub struct ReActLoop {
    council: ModelCouncil,
    dispatcher: ToolDispatcher,
    memory: Memory,
    parser: ThoughtParser,
    max_iterations: usize,
    agent_id: String,
}

impl ReActLoop {
    pub fn new(
        council: ModelCouncil,
        dispatcher: ToolDispatcher,
        memory: Memory,
        max_iterations: usize,
        agent_id: String,
    ) -> Self {
        Self {
            council,
            dispatcher,
            memory,
            parser: ThoughtParser::default(),
            max_iterations,
            agent_id,
        }
    }

    pub async fn run(&self, mut ctx: Context) -> Result<AgentResult, KairoError> {
        let mut thoughts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut total_usage = TokenUsage::default();

        ctx.messages.push(Message {
            role: Role::System,
            content: Self::build_system_prompt(),
            name: None,
            timestamp: chrono::Utc::now(),
        });

        for iteration in 0..self.max_iterations {
            debug!(iteration, agent_id = %self.agent_id, "ReAct loop iteration");

            for msg in self.memory.to_messages(5).await {
                ctx.messages.push(msg);
            }

            let options = CompletionOptions {
                temperature: None,
                max_tokens: None,
                ..Default::default()
            };

            let response = self.council.complete(/* task_type, messages, options */).await?;
            total_usage.prompt_tokens += response.usage.prompt_tokens;
            total_usage.completion_tokens += response.usage.completion_tokens;
            total_usage.total_tokens += response.usage.total_tokens;

            let parsed = self.parser.parse(&response.content)?;

            if parsed.action.as_deref() == Some("finish") {
                return Ok(AgentResult {
                    output: parsed.thought,
                    thoughts,
                    tool_calls,
                    token_usage: total_usage,
                });
            }

            if let Some(call) = parsed.tool_calls.first() {
                let result = self.dispatcher.dispatch(call).await?;
                let observation = format!("Tool '{}' returned: {}", call.tool_name, result.result);

                ctx.messages.push(Message {
                    role: Role::Assistant,
                    content: response.content.clone(),
                    name: None,
                    timestamp: chrono::Utc::now(),
                });
                ctx.messages.push(Message {
                    role: Role::Tool,
                    content: observation.clone(),
                    name: Some(call.tool_name.clone()),
                    timestamp: chrono::Utc::now(),
                });

                tool_calls.push(call.clone());
                thoughts.push(Thought {
                    thought: parsed.thought.clone(),
                    action: parsed.action.clone(),
                    observation: Some(observation),
                    tool_calls: parsed.tool_calls.clone(),
                });
            } else {
                thoughts.push(Thought {
                    thought: parsed.thought.clone(),
                    action: parsed.action.clone(),
                    observation: parsed.observation.clone(),
                    tool_calls: parsed.tool_calls.clone(),
                });
                ctx.messages.push(Message {
                    role: Role::Assistant,
                    content: response.content,
                    name: None,
                    timestamp: chrono::Utc::now(),
                });
            }
        }

        Ok(AgentResult {
            output: "Max iterations reached".into(),
            thoughts,
            tool_calls,
            token_usage: total_usage,
        })
    }

    fn build_system_prompt() -> String {
        "You are a ReAct agent. Think step by step.\n\
Available tools: calculator, web_search, filesystem.\n\
Use format:\n\
Thought: <reasoning>\n\
Action: <tool_name> or finish\n\
Tool: <tool_name>(arg1=value1,arg2=value2)\n\
Observation: <result>".to_string()
    }
}
```

Note: `ModelCouncil::complete` signature may need adjustment; inspect current API and match it. Pass `&kairo_core::types::TaskType::Reasoning` or equivalent.

- [ ] **Step 4.6: Refactor `ReActAgent` to delegate to `ReActLoop`**

In `kairo-agents/src/lib.rs`, keep the public `ReActAgent` struct but make `run` delegate:

```rust
pub async fn run(&self, ctx: Context) -> Result<AgentResult, KairoError> {
    with_agent_span(&self.agent.id.to_string(), async {
        let loop_ = ReActLoop::new(
            self.council.clone(),
            ToolDispatcher::new(self.tools.clone()),
            self.memory.clone(),
            self.max_iterations,
            self.agent.id.to_string(),
        );
        loop_.run(ctx).await
    }).await
}
```

- [ ] **Step 4.7: Run tests and fix compilation**

Run: `cargo test -p kairo-agents`
Expected: PASS.

- [ ] **Step 4.8: Commit**

```bash
git add kairo-agents/src
git commit -m "feat(agents): split ReActAgent into loop, thought parser, and tool dispatcher"
```

---

## Task 5: Split `WorkflowEngine` into DAG builder, store, executor, and runner

**Files:**
- Create: `kairo-orchestrator/src/dag.rs`
- Create: `kairo-orchestrator/src/store.rs`
- Create: `kairo-orchestrator/src/runner.rs`
- Create: `kairo-orchestrator/src/executor.rs`
- Modify: `kairo-orchestrator/src/lib.rs`

**Context:**
`WorkflowEngine` builds the graph, stores state, executes tasks, and directly depends on `ReActAgent`.

**Steps:**

- [ ] **Step 5.1: Write failing test for DAG builder**

Create `kairo-orchestrator/src/dag.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use kairo_core::{Task, Workflow, WorkflowStatus};
    use uuid::Uuid;

    #[test]
    fn builds_dag_for_empty_workflow() {
        let workflow = Workflow {
            id: Uuid::new_v4(),
            name: "test".into(),
            tasks: vec![],
            subtasks: vec![],
            status: WorkflowStatus::Draft,
        };
        let dag = DagBuilder::build(&workflow).unwrap();
        assert_eq!(dag.node_count(), 0);
    }
}
```

- [ ] **Step 5.2: Run and confirm failure**

Run: `cargo test -p kairo-orchestrator builds_dag_for_empty_workflow`
Expected: FAIL.

- [ ] **Step 5.3: Implement `DagBuilder`**

In `kairo-orchestrator/src/dag.rs`:

```rust
use kairo_core::{Subtask, Workflow};
use petgraph::graph::DiGraph;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct WorkflowDag {
    pub graph: DiGraph<Uuid, ()>,
    pub node_map: HashMap<Uuid, petgraph::graph::NodeIndex>,
}

pub struct DagBuilder;

impl DagBuilder {
    pub fn build(workflow: &Workflow) -> Result<WorkflowDag, kairo_core::error::KairoError> {
        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();

        for task in &workflow.tasks {
            let idx = graph.add_node(task.id);
            node_map.insert(task.id, idx);
        }

        for subtask in &workflow.subtasks {
            let &from = node_map
                .get(&subtask.parent_id)
                .ok_or_else(|| kairo_core::error::KairoError::Workflow(format!("Unknown parent task: {}", subtask.parent_id)))?;
            for dep_id in &subtask.dependencies {
                let &to = node_map
                    .get(dep_id)
                    .ok_or_else(|| kairo_core::error::KairoError::Workflow(format!("Unknown dependency: {}", dep_id)))?;
                graph.add_edge(to, from, ());
            }
        }

        Ok(WorkflowDag { graph, node_map })
    }
}
```

- [ ] **Step 5.4: Implement `WorkflowStore` seam**

In `kairo-orchestrator/src/store.rs`:

```rust
use async_trait::async_trait;
use kairo_core::error::KairoError;
use kairo_core::{Workflow, WorkflowStatus};
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Default, Clone)]
pub struct TaskState {
    pub status: kairo_core::TaskStatus,
    pub output: Option<String>,
}

#[derive(Debug, Default)]
pub struct WorkflowRecord {
    pub workflow: Workflow,
    pub statuses: HashMap<Uuid, TaskState>,
    pub overall_status: WorkflowStatus,
}

#[async_trait]
pub trait WorkflowStore: Send + Sync {
    async fn create(&self, workflow: Workflow) -> Result<(), KairoError>;
    async fn get(&self, id: Uuid) -> Result<Option<WorkflowRecord>, KairoError>;
    async fn update_task(&self, workflow_id: Uuid, task_id: Uuid, state: TaskState) -> Result<(), KairoError>;
    async fn set_status(&self, workflow_id: Uuid, status: WorkflowStatus) -> Result<(), KairoError>;
}

pub struct InMemoryWorkflowStore {
    records: RwLock<HashMap<Uuid, WorkflowRecord>>,
}

impl InMemoryWorkflowStore {
    pub fn new() -> Self {
        Self { records: RwLock::new(HashMap::new()) }
    }
}

#[async_trait]
impl WorkflowStore for InMemoryWorkflowStore {
    async fn create(&self, workflow: Workflow) -> Result<(), KairoError> {
        let mut records = self.records.write().await;
        let mut statuses = HashMap::new();
        for task in &workflow.tasks {
            statuses.insert(task.id, TaskState::default());
        }
        records.insert(workflow.id, WorkflowRecord {
            workflow,
            statuses,
            overall_status: WorkflowStatus::Draft,
        });
        Ok(())
    }

    async fn get(&self, id: Uuid) -> Result<Option<WorkflowRecord>, KairoError> {
        let records = self.records.read().await;
        Ok(records.get(&id).cloned())
    }

    async fn update_task(&self, workflow_id: Uuid, task_id: Uuid, state: TaskState) -> Result<(), KairoError> {
        let mut records = self.records.write().await;
        let record = records
            .get_mut(&workflow_id)
            .ok_or_else(|| KairoError::Workflow(format!("Workflow {} not found", workflow_id)))?;
        record.statuses.insert(task_id, state);
        Ok(())
    }

    async fn set_status(&self, workflow_id: Uuid, status: WorkflowStatus) -> Result<(), KairoError> {
        let mut records = self.records.write().await;
        let record = records
            .get_mut(&workflow_id)
            .ok_or_else(|| KairoError::Workflow(format!("Workflow {} not found", workflow_id)))?;
        record.overall_status = status;
        Ok(())
    }
}
```

- [ ] **Step 5.5: Implement `TaskRunner` seam**

In `kairo-orchestrator/src/runner.rs`:

```rust
use async_trait::async_trait;
use kairo_core::context::Context;
use kairo_core::error::KairoError;
use kairo_core::types::AgentResult;

#[async_trait]
pub trait TaskRunner: Send + Sync {
    async fn run(&self, ctx: Context) -> Result<AgentResult, KairoError>;
}
```

Keep an adapter for `ReActAgent` if needed, or make `WorkflowEngine` accept any `Arc<dyn TaskRunner>`.

- [ ] **Step 5.6: Implement `Executor`**

In `kairo-orchestrator/src/executor.rs`:

```rust
use kairo_core::context::Context;
use kairo_core::error::KairoError;
use kairo_core::types::{AgentResult, WorkflowResult, WorkflowStatus};
use petgraph::algo::toposort;
use std::sync::Arc;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::dag::WorkflowDag;
use crate::runner::TaskRunner;
use crate::store::{TaskState, WorkflowStore};

pub struct Executor<S: WorkflowStore, R: TaskRunner> {
    store: S,
    runner: Arc<R>,
}

impl<S: WorkflowStore, R: TaskRunner> Executor<S, R> {
    pub fn new(store: S, runner: Arc<R>) -> Self {
        Self { store, runner }
    }

    pub async fn execute(
        &self,
        workflow_id: Uuid,
        dag: WorkflowDag,
        ctx: Context,
    ) -> Result<WorkflowResult, KairoError> {
        let order = toposort(&dag.graph, None)
            .map_err(|_| KairoError::Workflow("Workflow contains a dependency cycle".into()))?;

        for node_idx in order {
            let task_id = dag.graph[node_idx];
            debug!(task_id = %task_id, "executing workflow task");

            self.store
                .update_task(workflow_id, task_id, TaskState { status: kairo_core::TaskStatus::InProgress, output: None })
                .await?;

            let mut task_ctx = ctx.clone();
            task_ctx.workflow_id = Some(workflow_id);

            match self.runner.run(task_ctx).await {
                Ok(result) => {
                    self.store
                        .update_task(workflow_id, task_id, TaskState { status: kairo_core::TaskStatus::Completed, output: Some(result.output.clone()) })
                        .await?;
                    info!(task_id = %task_id, "task completed");
                }
                Err(e) => {
                    self.store
                        .update_task(workflow_id, task_id, TaskState { status: kairo_core::TaskStatus::Failed, output: None })
                        .await?;
                    self.store.set_status(workflow_id, WorkflowStatus::Failed).await?;
                    error!(task_id = %task_id, error = %e, "task failed");
                    return Err(KairoError::Workflow(format!("Task {} failed: {}", task_id, e)));
                }
            }
        }

        self.store.set_status(workflow_id, WorkflowStatus::Completed).await?;
        Ok(WorkflowResult {
            workflow_id,
            outputs: HashMap::new(), // collect from store
            status: WorkflowStatus::Completed,
        })
    }
}
```

Add `use std::collections::HashMap;`.

- [ ] **Step 5.7: Refactor `WorkflowEngine`**

In `kairo-orchestrator/src/lib.rs`, keep the public struct but delegate:

```rust
pub struct WorkflowEngine {
    store: InMemoryWorkflowStore,
}

impl WorkflowEngine {
    pub fn new() -> Self {
        Self { store: InMemoryWorkflowStore::new() }
    }

    pub async fn register(&self, workflow: Workflow) -> Result<(), KairoError> {
        self.store.create(workflow).await
    }

    pub async fn execute(&self, workflow_id: Uuid, ctx: Context, runner: Arc<dyn TaskRunner>) -> Result<WorkflowResult, KairoError> {
        let record = self.store.get(workflow_id).await?
            .ok_or_else(|| KairoError::Workflow(format!("Workflow {} not found", workflow_id)))?;
        let dag = DagBuilder::build(&record.workflow)?;
        let executor = Executor::new(self.store.clone(), runner);
        executor.execute(workflow_id, dag, ctx).await
    }

    pub async fn get_status(&self, workflow_id: Uuid) -> Option<WorkflowStatus> {
        self.store.get(workflow_id).await.ok().flatten().map(|r| r.overall_status)
    }
}
```

Note: adjust `register` signature to drop `agents: HashMap<Uuid, Arc<ReActAgent>>`; the runner is now supplied at `execute` time. This is a public API change. Update callers in `kairo-cli/src/lib.rs` and `kairo-api/src/lib.rs`.

- [ ] **Step 5.8: Update callers**

In `kairo-cli/src/lib.rs` and `kairo-api/src/lib.rs`, when calling `engine.execute(...)`, pass an `Arc<dyn TaskRunner>` adapter wrapping the chosen `ReActAgent`.

- [ ] **Step 5.9: Run tests**

Run: `cargo test -p kairo-orchestrator`
Expected: PASS.

- [ ] **Step 5.10: Commit**

```bash
git add kairo-orchestrator/src kairo-cli/src/lib.rs kairo-api/src/lib.rs
git commit -m "feat(orchestrator): split workflow engine into DAG builder, store, executor, and runner seam"
```

---

## Task 6: Explicit `Telemetry` seam

**Files:**
- Create: `kairo-telemetry/src/interface.rs`
- Create: `kairo-telemetry/src/test_telemetry.rs`
- Modify: `kairo-telemetry/src/lib.rs`
- Modify: `kairo-cli/src/main.rs`
- Modify: `kairo-cli/src/lib.rs`
- Modify: `kairo-api/src/lib.rs`

**Context:**
Telemetry uses global `OnceLock` state. Tests can interfere, and there is no seam to swap in a test recorder.

**Steps:**

- [ ] **Step 6.1: Write failing test for `TestTelemetry`**

Create `kairo-telemetry/src/interface.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_records_counter() {
        let telemetry = TestTelemetry::new();
        telemetry.counter("requests", 1);
        assert_eq!(telemetry.get_counter("requests"), Some(1));
    }
}
```

- [ ] **Step 6.2: Run and confirm failure**

Run: `cargo test -p kairo-telemetry test_telemetry_records_counter`
Expected: FAIL.

- [ ] **Step 6.3: Implement `Telemetry` trait and `TestTelemetry`**

In `kairo-telemetry/src/interface.rs`:

```rust
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

pub trait Telemetry: Send + Sync {
    fn counter(&self, name: &'static str, value: u64);
    fn gauge(&self, name: &'static str, value: f64);
    fn histogram(&self, name: &'static str, value: f64);
}

pub struct TestTelemetry {
    counters: Arc<Mutex<HashMap<String, u64>>>,
    gauges: Arc<Mutex<HashMap<String, f64>>>,
    histograms: Arc<Mutex<HashMap<String, Vec<f64>>>>,
}

impl TestTelemetry {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(Mutex::new(HashMap::new())),
            gauges: Arc::new(Mutex::new(HashMap::new())),
            histograms: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn get_counter(&self, name: &str) -> Option<u64> {
        self.counters.lock().unwrap().get(name).copied()
    }
}

impl Telemetry for TestTelemetry {
    fn counter(&self, name: &'static str, value: u64) {
        let mut counters = self.counters.lock().unwrap();
        *counters.entry(name.to_string()).or_insert(0) += value;
    }

    fn gauge(&self, name: &'static str, value: f64) {
        self.gauges.lock().unwrap().insert(name.to_string(), value);
    }

    fn histogram(&self, name: &'static str, value: f64) {
        self.histograms.lock().unwrap().entry(name.to_string()).or_default().push(value);
    }
}
```

Also implement `PrometheusTelemetry` or `NoOpTelemetry` if needed. For this task, keep the global convenience but make it optional.

- [ ] **Step 6.4: Refactor `kairo-telemetry/src/lib.rs`**

Keep a global initializer for backward compatibility but add a builder that returns `Arc<dyn Telemetry>`:

```rust
pub mod interface;
pub use interface::{Telemetry, TestTelemetry};

pub fn build_default() -> Arc<dyn Telemetry> {
    Arc::new(TestTelemetry::new()) // or real implementation
}
```

- [ ] **Step 6.5: Wire telemetry into CLI and API**

In `kairo-cli/src/lib.rs`, add a `telemetry: Arc<dyn Telemetry>` field to `ApiState` if applicable, or pass it through. Keep it minimal: initialize in `main` and pass down.

In `kairo-api/src/lib.rs`, if `ApiState` exists, add a telemetry field and use it in handlers.

- [ ] **Step 6.6: Run tests**

Run: `cargo test -p kairo-telemetry -p kairo-cli -p kairo-api`
Expected: PASS.

- [ ] **Step 6.7: Commit**

```bash
git add kairo-telemetry/src kairo-cli/src kairo-api/src/lib.rs
git commit -m "feat(telemetry): introduce Telemetry seam and test adapter"
```

---

## Final verification

- [ ] **Run full workspace check**

```bash
cargo check --workspace --all-targets --all-features
```

Expected: exit 0 (warnings are OK).

- [ ] **Run full workspace tests**

```bash
cargo test --workspace
```

Expected: all tests pass.

- [ ] **Run clippy**

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Expected: exit 0.

---

## Self-review

1. **Spec coverage:** Each of the six deepening opportunities from the architecture report maps to one task.
2. **Placeholder scan:** Every step contains exact file paths and commands. Code blocks show real implementation shapes; adjust to exact current signatures after reading files.
3. **Type consistency:** `KairoError::Provider` now carries `ProviderError`; all provider adapters, API handlers, and workflow code must be updated consistently.
