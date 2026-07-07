use std::sync::Arc;

use kairo_core::{CompletionOptions, Context, KairoError, Message, Role, TaskType, TokenUsage};

use crate::{AgentResult, Thought};
use kairo_council::ModelCouncil;
use kairo_memory::HybridMemory;
use tracing::{debug, info};

use crate::thought_parser::ThoughtParser;
use crate::tool_dispatcher::ToolDispatcher;

pub struct ReActLoop {
    council: Arc<ModelCouncil>,
    dispatcher: ToolDispatcher,
    memory: Arc<HybridMemory>,
    parser: ThoughtParser,
    max_iterations: usize,
    agent_id: String,
    temperature: f32,
    max_tokens: u32,
}

impl ReActLoop {
    pub fn new(
        council: Arc<ModelCouncil>,
        dispatcher: ToolDispatcher,
        memory: Arc<HybridMemory>,
        max_iterations: usize,
        agent_id: String,
        temperature: f32,
        max_tokens: u32,
    ) -> Self {
        Self {
            council,
            dispatcher,
            memory,
            parser: ThoughtParser::default(),
            max_iterations,
            agent_id,
            temperature,
            max_tokens,
        }
    }

    pub async fn run(&self, mut ctx: Context) -> Result<AgentResult, KairoError> {
        let mut thoughts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut total_usage = TokenUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };

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
                temperature: Some(self.temperature),
                max_tokens: Some(self.max_tokens),
                ..Default::default()
            };

            let response = self
                .council
                .complete(&TaskType::Reasoning, ctx.messages.clone(), options)
                .await?;

            total_usage.prompt_tokens += response.usage.prompt_tokens;
            total_usage.completion_tokens += response.usage.completion_tokens;
            total_usage.total_tokens += response.usage.total_tokens;

            let parsed = self.parser.parse(&response.content)?;

            if parsed.action.as_deref() == Some("finish") {
                info!(agent_id = %self.agent_id, "ReAct loop finishing");
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
        "You are a ReAct agent. Think step by step. \
Available tools: calculator, web_search, filesystem. \
Use format: Thought: <your reasoning>\nAction: <tool_name> or finish\nTool: <tool_name>(arg1=value1,arg2=value2)\nObservation: <result>"
            .to_string()
    }
}
