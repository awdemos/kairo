use thiserror::Error;

use crate::provider_error::ProviderError;

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
