use clap::{Parser, Subcommand};
use std::sync::Arc;
use tracing::{info, warn};

use kairo_core::{Agent, AgentConfig, Context, KairoError, Message, ModelId, Role, TaskType};
use kairo_agents::ReActAgent;
use kairo_api::{ApiState, run_server};
use kairo_council::{default_council, bootstrap_council};
use kairo_memory::HybridMemory;
use kairo_tools::default_registry;

#[derive(Parser)]
#[command(name = "kairo")]
#[command(about = "Kairo - Agentic AI Orchestrator CLI")]
#[command(version = "0.1.0")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Chat {
        #[arg(short, long)]
        model: Option<String>,
        #[arg(short, long)]
        message: String,
    },
    Run {
        #[arg(short, long)]
        workflow: String,
    },
    Agent {
        #[arg(short, long)]
        task: String,
        #[arg(short, long, default_value = "gpt-4o")]
        model: String,
    },
    Server {
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },
}

pub async fn run(cli: Cli) -> Result<(), KairoError> {
    match cli.command {
        Commands::Chat { model, message } => {
            let model_id = parse_model(model.as_deref().unwrap_or("gpt-4o"));
            info!(model = ?model_id, "starting chat session");

            let api_key = get_api_key(&model_id)?;
            let council = Arc::new(default_council(&api_key));
            bootstrap_council(&council).await;

            let _ctx = Context::new(uuid::Uuid::new_v4());
            let messages = vec![
                Message {
                    role: Role::System,
                    content: "You are a helpful assistant.".into(),
                    name: None,
                    timestamp: chrono::Utc::now(),
                },
                Message {
                    role: Role::User,
                    content: message,
                    name: None,
                    timestamp: chrono::Utc::now(),
                },
            ];

            let options = kairo_core::CompletionOptions {
                temperature: Some(0.7),
                max_tokens: Some(2048),
                ..Default::default()
            };

            let response = council.complete(&TaskType::Reasoning, messages, options).await?;

            println!("\n{}", response.content);
            info!(tokens = response.usage.total_tokens, "response received");
            Ok(())
        }
        Commands::Run { workflow } => {
            info!(workflow = %workflow, "running workflow");
            println!("Workflow execution not yet implemented. Use 'agent' for single-task execution.");
            Ok(())
        }
        Commands::Agent { task, model } => {
            let model_id = parse_model(&model);
            info!(task = %task, model = ?model_id, "running agent task");

            let api_key = get_api_key(&model_id)?;
            let council = Arc::new(default_council(&api_key));
            bootstrap_council(&council).await;

            let tools = Arc::new(default_registry().await);
            let memory = Arc::new(HybridMemory::new());

            let agent = Agent {
                id: uuid::Uuid::new_v4(),
                model: model_id,
                config: AgentConfig::default(),
            };

            let react_agent = ReActAgent::new(agent, memory, tools, council);
            let ctx = Context::new(uuid::Uuid::new_v4());

            println!("\nTask: {}", task);
            println!("Running ReAct agent...\n");

            let result = react_agent.run(ctx).await?;

            println!("\n=== Result ===");
            println!("{}", result.output);
            println!("\n=== Token Usage ===");
            println!("Prompt: {}", result.token_usage.prompt_tokens);
            println!("Completion: {}", result.token_usage.completion_tokens);
            println!("Total: {}", result.token_usage.total_tokens);

            if !result.thoughts.is_empty() {
                println!("\n=== Reasoning Chain ===");
                for (i, thought) in result.thoughts.iter().enumerate() {
                    println!("\n[Step {}]", i + 1);
                    println!("Thought: {}", thought.thought);
                    if let Some(action) = &thought.action {
                        println!("Action: {}", action);
                    }
                    if let Some(obs) = &thought.observation {
                        println!("Observation: {}", obs);
                    }
                }
            }

            Ok(())
        }
        Commands::Server { port } => {
            info!(port, "starting API server");

            let api_key = std::env::var("OPENAI_API_KEY")
                .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
                .or_else(|_| std::env::var("GEMINI_API_KEY"))
                .unwrap_or_default();

            if api_key.is_empty() {
                warn!("No API key found in environment. Server will fail on chat requests.");
                warn!("Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY.");
            }

            let council = Arc::new(default_council(&api_key));
            bootstrap_council(&council).await;

            let tools = Arc::new(default_registry().await);
            let memory = Arc::new(HybridMemory::new());

            let agent = Agent {
                id: uuid::Uuid::new_v4(),
                model: ModelId::Gpt4o,
                config: AgentConfig::default(),
            };

            let react_agent = ReActAgent::new(agent, memory, tools, council);
            let agents = Arc::new(tokio::sync::RwLock::new(vec![react_agent]));

            let state = ApiState {
                agents,
                engine: Arc::new(kairo_orchestrator::WorkflowEngine::new()),
            };

            println!("Kairo API server running on http://0.0.0.0:{}", port);
            run_server(state, port).await
        }
    }
}

fn parse_model(name: &str) -> ModelId {
    match name {
        "gpt-4o" => ModelId::Gpt4o,
        "gpt-4o-mini" => ModelId::Gpt4oMini,
        "claude-3-5-sonnet" => ModelId::Claude3_5Sonnet,
        "claude-3-opus" => ModelId::Claude3Opus,
        "gemini-2.0-flash" => ModelId::Gemini2_0Flash,
        "gemini-1.5-pro" => ModelId::Gemini1_5Pro,
        "grok-2" => ModelId::Grok2,
        "llama-3-70b" => ModelId::Llama3_70b,
        "mistral-large" => ModelId::MistralLarge,
        _ => ModelId::Custom(name.to_string()),
    }
}

fn get_api_key(model: &ModelId) -> Result<String, KairoError> {
    let env_vars = match model {
        ModelId::Gpt4o | ModelId::Gpt4oMini | ModelId::Gpt4 | ModelId::Gpt4Turbo | ModelId::Gpt3_5Turbo | ModelId::O1 | ModelId::O3Mini => {
            vec!["OPENAI_API_KEY"]
        }
        ModelId::Claude3_5Sonnet | ModelId::Claude3Opus | ModelId::Claude3Haiku | ModelId::Claude3_5Haiku => {
            vec!["ANTHROPIC_API_KEY"]
        }
        ModelId::Gemini2_0Flash | ModelId::Gemini1_5Pro | ModelId::Gemini1_5Flash => {
            vec!["GEMINI_API_KEY"]
        }
        _ => {
            vec!["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
        }
    };

    for var in &env_vars {
        if let Ok(key) = std::env::var(var) {
            if !key.is_empty() {
                return Ok(key);
            }
        }
    }

    Err(KairoError::Unauthorized)
}
