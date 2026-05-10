use clap::{Parser, Subcommand};
use std::sync::Arc;
use tracing::{error, info};

use kairo_core::{Agent, AgentConfig, Context, KairoError, Message, ModelId, Role};
use kairo_agents::ReActAgent;
use kairo_orchestrator::WorkflowEngine;

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

            let agent = Agent {
                id: uuid::Uuid::new_v4(),
                model: model_id,
                config: AgentConfig::default(),
            };

            let ctx = Context::new(uuid::Uuid::new_v4());
            info!("User: {}", message);
            info!("Agent: Placeholder response - integrate with ReActAgent");
            Ok(())
        }
        Commands::Run { workflow } => {
            info!(workflow = %workflow, "running workflow");
            let engine = Arc::new(WorkflowEngine::new());
            info!("Workflow engine initialized");
            Ok(())
        }
        Commands::Agent { task } => {
            info!(task = %task, "running agent task");
            info!("Agent execution placeholder - integrate with ReActAgent");
            Ok(())
        }
        Commands::Server { port } => {
            info!(port, "starting API server");
            info!("Server placeholder - integrate with kairo-api");
            Ok(())
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
