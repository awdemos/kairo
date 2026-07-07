use std::sync::Arc;
use uuid::Uuid;

use kairo_core::{Agent, Context, KairoError, Message, Role};
use kairo_council::ModelCouncil;
use kairo_memory::HybridMemory;
use kairo_telemetry::with_agent_span;
use kairo_tools::ToolRegistry;

pub mod react_loop;
pub mod thought_parser;
pub mod tool_dispatcher;

pub use react_loop::ReActLoop;
pub use thought_parser::{ParsedThought, ThoughtParser};
pub use tool_dispatcher::ToolDispatcher;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Thought {
    pub thought: String,
    pub action: Option<String>,
    pub observation: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ToolCall {
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentResult {
    pub output: String,
    pub thoughts: Vec<Thought>,
    pub tool_calls: Vec<ToolCall>,
    pub token_usage: kairo_core::TokenUsage,
}

pub struct ReActAgent {
    agent: Agent,
    memory: Arc<HybridMemory>,
    tools: Arc<ToolRegistry>,
    council: Arc<ModelCouncil>,
    max_iterations: usize,
}

impl std::fmt::Debug for ReActAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReActAgent")
            .field("agent", &self.agent)
            .field("max_iterations", &self.max_iterations)
            .finish_non_exhaustive()
    }
}

impl ReActAgent {
    pub fn new(
        agent: Agent,
        memory: Arc<HybridMemory>,
        tools: Arc<ToolRegistry>,
        council: Arc<ModelCouncil>,
    ) -> Self {
        Self {
            agent,
            memory,
            tools,
            council,
            max_iterations: 10,
        }
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub async fn run(&self, ctx: Context) -> Result<AgentResult, KairoError> {
        with_agent_span(&self.agent.id.to_string(), async {
            let loop_ = ReActLoop::new(
                Arc::clone(&self.council),
                ToolDispatcher::new((*self.tools).clone()),
                Arc::clone(&self.memory),
                self.max_iterations,
                self.agent.id.to_string(),
                self.agent.config.temperature,
                self.agent.config.max_tokens,
            );
            loop_.run(ctx).await
        })
        .await
    }
}

pub struct SubAgent {
    parent_id: Uuid,
    agent: ReActAgent,
}

impl SubAgent {
    pub fn new(parent_id: Uuid, agent: ReActAgent) -> Self {
        Self { parent_id, agent }
    }

    pub async fn decompose(&self, task: &str, ctx: Context) -> Result<Vec<String>, KairoError> {
        let prompt = format!("Decompose the following task into subtasks:\n{}", task);
        let mut ctx = ctx;
        ctx.messages.push(Message {
            role: Role::User,
            content: prompt,
            name: None,
            timestamp: chrono::Utc::now(),
        });

        let result = self.agent.run(ctx).await?;
        let subtasks: Vec<String> = result
            .output
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        Ok(subtasks)
    }
}
