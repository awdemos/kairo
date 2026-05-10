use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::Message;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub session_id: Uuid,
    pub agent_id: Option<Uuid>,
    pub workflow_id: Option<Uuid>,
    pub messages: Vec<Message>,
    pub metadata: serde_json::Value,
    pub depth: u32,
}

impl Context {
    pub fn new(session_id: Uuid) -> Self {
        Self {
            session_id,
            agent_id: None,
            workflow_id: None,
            messages: Vec::new(),
            metadata: serde_json::Value::Null,
            depth: 0,
        }
    }
}
