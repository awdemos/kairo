pub mod agent;
pub mod config;
pub mod context;
pub mod error;
pub mod model;
pub mod model_id;
pub mod provider_error;
pub mod task;
pub mod traits;
pub mod types;
pub mod workflow;

pub use agent::{Agent, AgentConfig};
pub use config::{
    ApiConfig, KairoConfig, ProviderConfig, TelemetryConfig, ToolConfig,
};
pub use context::Context;
pub use error::KairoError;
pub use model::ModelId;
pub use provider_error::{ProviderError, ProviderIdentity};
pub use task::TaskType;
pub use traits::{Connector, Executable, Provider, Routable, Tool};
pub use types::{
    Action, CompletionOptions, CompletionResponse, CostEstimate, Data, Event, LatencyTarget,
    Message, Output, Role, TaskVector, TokenUsage, ToolInput, ToolOutput,
};
pub use workflow::{Subtask, Task, TaskStatus, Workflow, WorkflowStatus};

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_model_id_variants() {
        let models = [
            ModelId::Gpt4o,
            ModelId::Claude3_5Sonnet,
            ModelId::Gemini2_0Flash,
            ModelId::Custom("test-model".into()),
        ];
        assert_eq!(models.len(), 4);
        assert!(matches!(models[0], ModelId::Gpt4o));
        assert!(matches!(models[3], ModelId::Custom(_)));
    }

    #[test]
    fn test_task_type_variants() {
        let tasks = [
            TaskType::Research,
            TaskType::CodeGeneration,
            TaskType::Custom("special".into()),
        ];
        assert_eq!(tasks.len(), 3);
        assert!(matches!(tasks[0], TaskType::Research));
    }

    #[test]
    fn test_kairo_error_display() {
        let err = KairoError::Provider(
            ProviderError::new("openai", "gpt-4o").with_message("test error"),
        );
        assert_eq!(
            err.to_string(),
            "Provider error: openai (gpt-4o) [no status]: test error"
        );
    }

    #[test]
    fn test_context_creation() {
        let ctx = Context::new(Uuid::new_v4());
        assert_eq!(ctx.depth, 0);
        assert!(ctx.agent_id.is_none());
    }

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.max_tokens, 4096);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.retry_count, 3);
    }

    #[test]
    fn test_role_equality() {
        assert_eq!(Role::User, Role::User);
        assert_ne!(Role::User, Role::Assistant);
    }

    #[test]
    fn test_token_usage_creation() {
        let usage = TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };
        assert_eq!(usage.total_tokens, 30);
    }
}
