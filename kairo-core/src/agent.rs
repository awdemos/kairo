use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::model::ModelId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub max_tokens: u32,
    pub temperature: f32,
    pub system_prompt: Option<String>,
    pub timeout_seconds: u64,
    pub retry_count: u32,
    pub custom_params: serde_json::Value,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            temperature: 0.7,
            system_prompt: None,
            timeout_seconds: 120,
            retry_count: 3,
            custom_params: serde_json::Value::Null,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: Uuid,
    pub model: ModelId,
    pub config: AgentConfig,
}
