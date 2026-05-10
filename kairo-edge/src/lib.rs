use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use kairo_core::{CompletionOptions, CompletionResponse, KairoError, Message, ModelId};

#[wasm_bindgen]
pub struct EdgeAgent {
    model: String,
    api_key: String,
}

#[wasm_bindgen]
impl EdgeAgent {
    #[wasm_bindgen(constructor)]
    pub fn new(model: String, api_key: String) -> Self {
        Self { model, api_key }
    }

    #[wasm_bindgen]
    pub async fn complete(&self, messages: JsValue) -> Result<JsValue, JsValue> {
        let msgs: Vec<EdgeMessage> = serde_wasm_bindgen::from_value(messages)
            .map_err(|e| JsValue::from_str(&format!("Parse error: {}", e)))?;

        let core_messages: Vec<Message> = msgs.into_iter().map(|m| Message {
            role: match m.role.as_str() {
                "system" => kairo_core::Role::System,
                "assistant" => kairo_core::Role::Assistant,
                "tool" => kairo_core::Role::Tool,
                _ => kairo_core::Role::User,
            },
            content: m.content,
            name: None,
            timestamp: chrono::Utc::now(),
        }).collect();

        let response = EdgeResponse {
            content: format!("Placeholder response from edge agent using {}", self.model),
            model: self.model.clone(),
        };

        serde_wasm_bindgen::to_value(&response)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EdgeMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EdgeResponse {
    pub content: String,
    pub model: String,
}

#[wasm_bindgen]
pub fn init_edge() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}
