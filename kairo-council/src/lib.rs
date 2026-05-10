use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, instrument};

use kairo_core::{CompletionOptions, CompletionResponse, KairoError, Message, ModelId, Provider, TaskType};
use kairo_providers::create_provider;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityScore {
    pub model: ModelId,
    pub score: f32,
    pub latency_ms: u64,
    pub cost_per_1k: f32,
}

pub struct RoutingDecision {
    pub model: ModelId,
    pub provider: Arc<dyn Provider>,
    pub confidence: f32,
}

impl std::fmt::Debug for RoutingDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoutingDecision")
            .field("model", &self.model)
            .field("provider", &"<dyn Provider>")
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl Clone for RoutingDecision {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            provider: Arc::clone(&self.provider),
            confidence: self.confidence,
        }
    }
}

pub struct ModelCouncil {
    scores: RwLock<HashMap<ModelId, CapabilityScore>>,
    api_keys: HashMap<ModelId, String>,
}

impl ModelCouncil {
    pub fn new() -> Self {
        Self {
            scores: RwLock::new(HashMap::new()),
            api_keys: HashMap::new(),
        }
    }

    pub fn with_api_key(mut self, model: ModelId, key: impl Into<String>) -> Self {
        self.api_keys.insert(model, key.into());
        self
    }

    #[instrument(skip(self))]
    pub async fn register_model(&self, score: CapabilityScore) {
        debug!(model = ?score.model, score = score.score, "registering model capability");
        let mut scores = self.scores.write().await;
        scores.insert(score.model.clone(), score);
    }

    pub async fn route(&self, task: &TaskType, _options: &CompletionOptions) -> Result<RoutingDecision, KairoError> {
        let scores = self.scores.read().await;
        let candidates: Vec<_> = scores.values().collect();

        if candidates.is_empty() {
            return Err(KairoError::Routing("No models registered in council".into()));
        }

        let best = candidates
            .into_iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| KairoError::Routing("No suitable model found".into()))?;

        let api_key = self.api_keys.get(&best.model)
            .ok_or_else(|| KairoError::Routing(format!("No API key for model {:?}", best.model)))?;

        let provider = create_provider(&best.model, api_key)?;

        Ok(RoutingDecision {
            model: best.model.clone(),
            provider,
            confidence: best.score,
        })
    }

    pub async fn complete(&self, task: &TaskType, messages: Vec<Message>, options: CompletionOptions) -> Result<CompletionResponse, KairoError> {
        let decision = self.route(task, &options).await?;
        debug!(model = ?decision.model, "delegating to provider");
        decision.provider.as_ref().complete(messages, options).await
    }

    pub async fn list_models(&self) -> Vec<CapabilityScore> {
        let scores = self.scores.read().await;
        scores.values().cloned().collect()
    }
}

impl Default for ModelCouncil {
    fn default() -> Self {
        Self::new()
    }
}

pub fn default_council(api_key: &str) -> ModelCouncil {
    ModelCouncil::new()
        .with_api_key(ModelId::Gpt4o, api_key)
        .with_api_key(ModelId::Claude3_5Sonnet, api_key)
        .with_api_key(ModelId::Gemini2_0Flash, api_key)
}

pub async fn bootstrap_council(council: &ModelCouncil) {
    council.register_model(CapabilityScore {
        model: ModelId::Gpt4o,
        score: 0.95,
        latency_ms: 800,
        cost_per_1k: 0.005,
    }).await;
    council.register_model(CapabilityScore {
        model: ModelId::Claude3_5Sonnet,
        score: 0.93,
        latency_ms: 1200,
        cost_per_1k: 0.003,
    }).await;
    council.register_model(CapabilityScore {
        model: ModelId::Gemini2_0Flash,
        score: 0.90,
        latency_ms: 600,
        cost_per_1k: 0.0005,
    }).await;
}
