use std::sync::Arc;

use kairo_core::{KairoError, ModelId, Provider};

pub mod anthropic;
pub mod gemini;
pub mod openai;
pub mod registry;

pub use anthropic::AnthropicProvider;
pub use gemini::GeminiProvider;
pub use openai::OpenAiProvider;
pub use registry::{ProviderAdapter, ProviderRegistry};

pub fn create_provider(model: &ModelId, api_key: &str) -> Result<Arc<dyn Provider>, KairoError> {
    ProviderRegistry::with_defaults().resolve(model, api_key)
}

pub(crate) fn is_retryable(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503)
}
