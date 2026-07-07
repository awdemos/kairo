use std::sync::Arc;

use kairo_core::error::KairoError;
use kairo_core::model::ModelId;
use kairo_core::traits::Provider;

use crate::{AnthropicProvider, GeminiProvider, OpenAiProvider};

pub trait ProviderAdapter: Send + Sync {
    fn matches(&self, model: &ModelId) -> bool;
    fn build(&self, model: &ModelId, api_key: &str) -> Arc<dyn Provider>;
}

pub struct OpenAiAdapter;

impl ProviderAdapter for OpenAiAdapter {
    fn matches(&self, model: &ModelId) -> bool {
        let openai_known = matches!(
            model,
            ModelId::Gpt4o
                | ModelId::Gpt4oMini
                | ModelId::Gpt4
                | ModelId::Gpt4Turbo
                | ModelId::Gpt3_5Turbo
                | ModelId::O1
                | ModelId::O3Mini
        );
        if openai_known {
            return true;
        }
        if let ModelId::Custom(name) = model {
            if let Some(ModelId::Custom(_)) = ModelId::resolve(name) {
                return name.starts_with("gpt-")
                    || name.starts_with("o1")
                    || name.starts_with("o3")
                    || name.starts_with("o4");
            }
        }
        false
    }

    fn build(&self, model: &ModelId, api_key: &str) -> Arc<dyn Provider> {
        Arc::new(OpenAiProvider::new(api_key, model.to_string()))
    }
}

pub struct AnthropicAdapter;

impl ProviderAdapter for AnthropicAdapter {
    fn matches(&self, model: &ModelId) -> bool {
        let known = matches!(
            model,
            ModelId::Claude3_5Sonnet
                | ModelId::Claude3Opus
                | ModelId::Claude3Haiku
                | ModelId::Claude3_5Haiku
                | ModelId::Claude4
                | ModelId::Claude4Opus
        );
        if known {
            return true;
        }
        if let ModelId::Custom(name) = model {
            if let Some(ModelId::Custom(_)) = ModelId::resolve(name) {
                return name.starts_with("claude-");
            }
        }
        false
    }

    fn build(&self, model: &ModelId, api_key: &str) -> Arc<dyn Provider> {
        Arc::new(AnthropicProvider::new(api_key, model.to_string()))
    }
}

pub struct GeminiAdapter;

impl ProviderAdapter for GeminiAdapter {
    fn matches(&self, model: &ModelId) -> bool {
        let known = matches!(
            model,
            ModelId::Gemini2_0Flash
                | ModelId::Gemini2_0Pro
                | ModelId::Gemini2_5Flash
                | ModelId::Gemini2_5Pro
                | ModelId::Gemini1_5Pro
                | ModelId::Gemini1_5Flash
        );
        if known {
            return true;
        }
        if let ModelId::Custom(name) = model {
            if let Some(ModelId::Custom(_)) = ModelId::resolve(name) {
                return name.starts_with("gemini-");
            }
        }
        false
    }

    fn build(&self, model: &ModelId, api_key: &str) -> Arc<dyn Provider> {
        Arc::new(GeminiProvider::new(api_key, model.to_string()))
    }
}

pub struct ProviderRegistry {
    adapters: Vec<Box<dyn ProviderAdapter>>,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self { adapters: Vec::new() }
    }

    pub fn register<A: ProviderAdapter + 'static>(mut self, adapter: A) -> Self {
        self.adapters.push(Box::new(adapter));
        self
    }

    pub fn with_defaults() -> Self {
        Self::new()
            .register(OpenAiAdapter)
            .register(AnthropicAdapter)
            .register(GeminiAdapter)
    }

    pub fn resolve(
        &self,
        model: &ModelId,
        api_key: &str,
    ) -> Result<Arc<dyn Provider>, KairoError> {
        for adapter in &self.adapters {
            if adapter.matches(model) {
                return Ok(adapter.build(model, api_key));
            }
        }
        Err(KairoError::Model(format!(
            "No provider available for model: {:?}",
            model
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_resolves_openai_models() {
        let registry = ProviderRegistry::with_defaults();
        let provider = registry.resolve(&ModelId::Gpt4o, "key");
        assert!(provider.is_ok());
    }

    #[test]
    fn registry_resolves_known_custom_aliases() {
        let registry = ProviderRegistry::with_defaults();
        let provider = registry.resolve(&ModelId::Custom("gpt-custom".into()), "key");
        assert!(provider.is_ok());
    }

    #[test]
    fn registry_rejects_unknown_models() {
        let registry = ProviderRegistry::with_defaults();
        let provider = registry.resolve(&ModelId::Custom("totally-unknown".into()), "key");
        assert!(provider.is_err());
    }
}
