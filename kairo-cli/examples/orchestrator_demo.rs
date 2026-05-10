use std::collections::HashMap;
use std::sync::Arc;

use kairo_agents::ReActAgent;
use kairo_core::{Agent, AgentConfig, CompletionOptions, Context, Message, ModelId, Role, Subtask, Task, TaskStatus, TaskType, Workflow, WorkflowStatus};
use kairo_council::{bootstrap_council, default_council};
use kairo_memory::HybridMemory;
use kairo_orchestrator::WorkflowEngine;
use kairo_tools::default_registry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("kairo=info")
        .init();

    println!("=== Kairo Orchestrator Example ===\n");

    let api_key = std::env::var("OPENAI_API_KEY")
        .unwrap_or_else(|_| {
            println!("Note: Set OPENAI_API_KEY for live provider responses\n");
            String::new()
        });

    let council = Arc::new(default_council(&api_key));
    bootstrap_council(&council).await;

    let memory = Arc::new(HybridMemory::new());
    let tools = Arc::new(default_registry().await);

    println!("1. Simple Chat");
    println!("   Query: 'What is quantum computing?'");
    match council.complete(
        &TaskType::Research,
        vec![Message {
            role: Role::User,
            content: "What is quantum computing?".into(),
            name: None,
            timestamp: chrono::Utc::now(),
        }],
        CompletionOptions::default(),
    ).await {
        Ok(resp) => println!("   Response: {}\n", resp.content),
        Err(e) => println!("   (Provider unavailable: {} - set OPENAI_API_KEY for live responses)\n", e),
    }

    println!("2. Agent with Tools");
    println!("   Task: 'Calculate 2^10 and search for Rust news'");
    let agent = ReActAgent::new(
        Agent {
            id: uuid::Uuid::new_v4(),
            model: ModelId::Gpt4o,
            config: AgentConfig::default(),
        },
        memory.clone(),
        tools.clone(),
        council.clone(),
    );
    let ctx = Context::new(uuid::Uuid::new_v4());
    match agent.run(ctx).await {
        Ok(result) => println!("   Result: {}\n", result.output),
        Err(e) => println!("   (Agent execution: {} - set OPENAI_API_KEY for live responses)\n", e),
    }

    println!("3. Workflow Execution");
    println!("   Workflow: research -> analyze -> summarize");

    let task1_id = uuid::Uuid::new_v4();
    let task2_id = uuid::Uuid::new_v4();
    let task3_id = uuid::Uuid::new_v4();

    let workflow = Workflow {
        id: uuid::Uuid::new_v4(),
        name: "research-workflow".into(),
        tasks: vec![
            Task {
                id: task1_id,
                task_type: TaskType::Research,
                description: "Search for Rust async runtime benchmarks".into(),
                input: serde_json::json!({"query": "Rust async runtime benchmarks 2024"}),
                expected_output: None,
                assigned_model: Some(ModelId::Gpt4oMini),
            },
            Task {
                id: task2_id,
                task_type: TaskType::DataAnalysis,
                description: "Analyze benchmark results".into(),
                input: serde_json::json!({}),
                expected_output: None,
                assigned_model: Some(ModelId::Claude3_5Sonnet),
            },
            Task {
                id: task3_id,
                task_type: TaskType::Summarization,
                description: "Summarize findings".into(),
                input: serde_json::json!({}),
                expected_output: None,
                assigned_model: Some(ModelId::Gpt4o),
            },
        ],
        subtasks: vec![
            Subtask {
                id: task2_id,
                parent_id: task2_id,
                task_type: TaskType::DataAnalysis,
                description: "Analyze".into(),
                dependencies: vec![task1_id],
                status: TaskStatus::Pending,
            },
            Subtask {
                id: task3_id,
                parent_id: task3_id,
                task_type: TaskType::Summarization,
                description: "Summarize".into(),
                dependencies: vec![task2_id],
                status: TaskStatus::Pending,
            },
        ],
        status: WorkflowStatus::Draft,
    };

    let engine = WorkflowEngine::new();
    let mut agents = HashMap::new();

    let agent1 = Arc::new(ReActAgent::new(
        Agent {
            id: uuid::Uuid::new_v4(),
            model: ModelId::Gpt4oMini,
            config: AgentConfig::default(),
        },
        memory.clone(),
        tools.clone(),
        council.clone(),
    ));
    agents.insert(task1_id, agent1.clone());
    agents.insert(task2_id, agent1.clone());
    agents.insert(task3_id, agent1);

    engine.register(workflow, agents).await?;
    let ctx = Context::new(uuid::Uuid::new_v4());

    match engine.execute(task1_id, ctx).await {
        Ok(result) => {
            println!("   Workflow completed with {} outputs", result.outputs.len());
            for (id, output) in result.outputs {
                println!("   Task {}: {}", id, output);
            }
        }
        Err(e) => println!("   (Workflow execution: {} - set OPENAI_API_KEY for live responses)\n", e),
    }

    println!("\n=== Example Complete ===");
    println!("Set OPENAI_API_KEY environment variable for live LLM responses.");

    Ok(())
}
