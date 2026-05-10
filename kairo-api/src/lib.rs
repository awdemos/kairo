use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{sse::Event, Sse},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};
use uuid::Uuid;

use kairo_core::{Agent, AgentConfig, Context, KairoError, Message, ModelId, Role, Workflow};
use kairo_agents::ReActAgent;
use kairo_orchestrator::{WorkflowEngine, WorkflowResult};
use kairo_telemetry::init as init_telemetry;

#[derive(Clone)]
pub struct ApiState {
    pub agents: Arc<tokio::sync::RwLock<Vec<ReActAgent>>>,
    pub engine: Arc<WorkflowEngine>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ApiMessage>,
    pub model: Option<String>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub content: String,
    pub model: String,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkflowRequest {
    pub name: String,
    pub tasks: Vec<WorkflowTaskRequest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkflowTaskRequest {
    pub description: String,
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WorkflowResponse {
    pub workflow_id: String,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

pub fn app(state: ApiState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/workflows", post(create_workflow))
        .route("/v1/workflows/:id", get(get_workflow))
        .with_state(state)
}

async fn health_check() -> &'static str {
    "ok"
}

async fn chat_completions(
    State(state): State<ApiState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, Json<ErrorResponse>)> {
    let model = req.model.as_deref().unwrap_or("gpt-4o");

    let messages: Vec<Message> = req.messages.into_iter().map(|m| Message {
        role: match m.role.as_str() {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        },
        content: m.content,
        name: None,
        timestamp: chrono::Utc::now(),
    }).collect();

    let ctx = Context::new(Uuid::new_v4());

    let agents = state.agents.read().await;
    let agent = agents.first().ok_or_else(|| {
        (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse {
            error: "No agents available".into(),
        }))
    })?;

    let result = agent.run(ctx).await.map_err(|e| {
        error!(error = %e, "agent execution failed");
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: e.to_string(),
        }))
    })?;

    Ok(Json(ChatResponse {
        id: Uuid::new_v4().to_string(),
        content: result.output,
        model: model.to_string(),
        finish_reason: "stop".to_string(),
    }))
}

async fn create_workflow(
    State(state): State<ApiState>,
    Json(req): Json<WorkflowRequest>,
) -> Result<Json<WorkflowResponse>, (StatusCode, Json<ErrorResponse>)> {
    let workflow_id = Uuid::new_v4();
    let workflow = Workflow {
        id: workflow_id,
        name: req.name,
        tasks: Vec::new(),
        subtasks: Vec::new(),
        status: kairo_core::WorkflowStatus::Draft,
    };

    state.engine.register(workflow, HashMap::new()).await.map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: e.to_string(),
        }))
    })?;

    Ok(Json(WorkflowResponse {
        workflow_id: workflow_id.to_string(),
        status: "created".to_string(),
    }))
}

async fn get_workflow(
    State(state): State<ApiState>,
    Path(id): Path<Uuid>,
) -> Result<Json<WorkflowResult>, (StatusCode, Json<ErrorResponse>)> {
    let outputs = state.engine.get_outputs(id).await.ok_or_else(|| {
        (StatusCode::NOT_FOUND, Json(ErrorResponse {
            error: "Workflow not found".into(),
        }))
    })?;

    let status = state.engine.get_status(id).await.unwrap_or(kairo_core::WorkflowStatus::Failed);

    Ok(Json(WorkflowResult {
        workflow_id: id,
        outputs,
        status,
    }))
}

pub async fn run_server(state: ApiState, port: u16) -> Result<(), KairoError> {
    init_telemetry().map_err(|e| KairoError::Internal(e.to_string()))?;

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    info!("Starting Kairo API server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await
        .map_err(|e| KairoError::Internal(format!("Failed to bind: {}", e)))?;

    axum::serve(listener, app(state)).await
        .map_err(|e| KairoError::Internal(format!("Server error: {}", e)))?;

    Ok(())
}
