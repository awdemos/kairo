use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument};
use uuid::Uuid;

use kairo_core::{
    Agent, AgentConfig, CompletionOptions, CompletionResponse, Context, KairoError, Message, ModelId,
    Provider, Role, TaskType, Tool, ToolInput, ToolOutput,
};
use kairo_memory::HybridMemory;
use kairo_tools::ToolRegistry;
use kairo_council::{ModelCouncil, RoutingDecision};
use kairo_telemetry::{agent_span, with_agent_span};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    pub thought: String,
    pub action: Option<String>,
    pub observation: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool_name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    #[instrument(skip(self, ctx))]
    pub async fn run(&self, ctx: Context) -> Result<AgentResult, KairoError> {
        with_agent_span(&self.agent.id.to_string(), async {
            self.run_inner(ctx).await
        }).await
    }

    async fn run_inner(&self, mut ctx: Context) -> Result<AgentResult, KairoError> {
        let mut thoughts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut total_usage = kairo_core::TokenUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };

        let system_prompt = self.build_system_prompt();
        ctx.messages.push(Message {
            role: Role::System,
            content: system_prompt,
            name: None,
            timestamp: chrono::Utc::now(),
        });

        for iteration in 0..self.max_iterations {
            debug!(iteration, "ReAct loop iteration");

            let memory_messages = self.memory.to_messages(5).await;
            for msg in memory_messages {
                ctx.messages.push(msg);
            }

            let options = CompletionOptions {
                temperature: Some(self.agent.config.temperature),
                max_tokens: Some(self.agent.config.max_tokens),
                ..Default::default()
            };

            let response = self.council.complete(
                &TaskType::Reasoning,
                ctx.messages.clone(),
                options,
            ).await?;

            total_usage.prompt_tokens += response.usage.prompt_tokens;
            total_usage.completion_tokens += response.usage.completion_tokens;
            total_usage.total_tokens += response.usage.total_tokens;

            let thought = self.parse_thought(&response.content)?;

            if let Some(action) = &thought.action {
                if action == "finish" {
                    return Ok(AgentResult {
                        output: thought.thought.clone(),
                        thoughts,
                        tool_calls,
                        token_usage: total_usage,
                    });
                }
            }

            if let Some(tool_call) = thought.tool_calls.first() {
                let result = self.execute_tool(tool_call).await?;
                let observation = format!("Tool '{}' returned: {}", tool_call.tool_name, result.result);

                ctx.messages.push(Message {
                    role: Role::Assistant,
                    content: response.content.clone(),
                    name: None,
                    timestamp: chrono::Utc::now(),
                });
                ctx.messages.push(Message {
                    role: Role::Tool,
                    content: observation.clone(),
                    name: Some(tool_call.tool_name.clone()),
                    timestamp: chrono::Utc::now(),
                });

                tool_calls.push(tool_call.clone());
                thoughts.push(Thought {
                    thought: thought.thought.clone(),
                    action: thought.action.clone(),
                    observation: Some(observation),
                    tool_calls: thought.tool_calls.clone(),
                });
            } else {
                thoughts.push(thought);
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

    fn build_system_prompt(&self) -> String {
        format!(
            "You are a ReAct agent. Think step by step. \
Available tools: {}. \
Use format: Thought: <your reasoning>\nAction: <tool_name> or finish\nObservation: <result>",
            "calculator, web_search, filesystem"
        )
    }

    fn parse_thought(&self, content: &str) -> Result<Thought, KairoError> {
        let thought = content.lines().next().unwrap_or(content).to_string();
        Ok(Thought {
            thought,
            action: None,
            tool_calls: Vec::new(),
            observation: None,
        })
    }

    async fn execute_tool(&self, call: &ToolCall) -> Result<ToolOutput, KairoError> {
        let input = ToolInput {
            arguments: call.arguments.clone(),
        };
        self.tools.execute(&call.tool_name, input).await
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
        let subtasks: Vec<String> = result.output
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        Ok(subtasks)
    }
}
