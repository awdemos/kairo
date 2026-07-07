use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};
use uuid::Uuid;

use kairo_core::{Context, KairoError, Message, Role, Workflow};
use kairo_agents::ReActAgent;
use kairo_orchestrator::{WorkflowEngine, WorkflowResult};
use kairo_telemetry::{Telemetry, init as init_telemetry};

#[derive(Clone)]
pub struct ApiState {
    pub agents: Arc<tokio::sync::RwLock<Vec<ReActAgent>>>,
    pub engine: Arc<WorkflowEngine>,
    pub telemetry: Arc<dyn Telemetry>,
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
        .route("/v1/workflows/{id}", get(get_workflow))
        .with_state(state)
}

async fn health_check() -> &'static str {
    "ok"
}

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

async fn chat_completions(
    State(state): State<ApiState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, Json<ErrorResponse>)> {
    let model = req.model.as_deref().unwrap_or("gpt-4o");

    let _messages: Vec<Message> = req.messages.into_iter().map(|m| Message {
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

    let result = agent.run(ctx).await.map_err(into_response)?;

    state.telemetry.counter("api_chat_completions", 1);

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

    state.engine.register(workflow).await.map_err(into_response)?;

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
    let _guard = init_telemetry().map_err(|e| KairoError::Internal(e.to_string()))?;

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    info!("Starting Kairo API server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await
        .map_err(|e| KairoError::Internal(format!("Failed to bind: {}", e)))?;

    axum::serve(listener, app(state)).await
        .map_err(|e| KairoError::Internal(format!("Server error: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use kairo_core::ProviderError;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

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

    #[tokio::test]
    async fn test_health_check() {
        let state = ApiState {
            agents: Arc::new(RwLock::new(vec![])),
            engine: Arc::new(WorkflowEngine::new()),
            telemetry: kairo_telemetry::build_default(),
        };
        let app = app(state);
        let response = app
            .oneshot(axum::http::Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

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
}
