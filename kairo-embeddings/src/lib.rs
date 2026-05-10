//! # kairo-embeddings
//!
//! Embedding generation and vector math utilities for the Kairo agentic AI orchestrator.
//!
//! Provides:
//! - [`EmbeddingClient`] trait and implementations (`LocalEmbeddingClient`, `RemoteEmbeddingClient`)
//! - [`Embedding`] type wrapping a dense `f32` vector
//! - SIMD-friendly vector math utilities
//! - [`EmbeddingStore`] trait and an in-memory implementation

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};
use uuid::Uuid;

pub use kairo_core::KairoError;

/// A dense embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Embedding {
    /// Unique identifier for this embedding record.
    pub id: String,
    /// The raw vector components.
    pub vector: Vec<f32>,
    /// Optional metadata (e.g. source text, model name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<EmbeddingMetadata>,
}

impl Eq for Embedding {}

impl std::hash::Hash for Embedding {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// Optional metadata attached to an embedding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingMetadata {
    /// Original text that produced this embedding.
    pub text: String,
    /// Name of the model used.
    pub model: String,
    /// Timestamp of creation (RFC 3339).
    pub created_at: String,
}

impl Embedding {
    /// Create a new embedding with the given vector and auto-generated id.
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            vector,
            metadata: None,
        }
    }

    /// Create a new embedding with explicit id.
    pub fn with_id(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata: None,
        }
    }

    /// Attach metadata.
    pub fn with_metadata(mut self, metadata: EmbeddingMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Dimensionality of the vector.
    pub fn dim(&self) -> usize {
        self.vector.len()
    }
}

/// Compute the dot product of two `f32` slices.
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot_product: vectors must have the same length"
    );
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the L2 (Euclidean) norm of a vector.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in the range `[-1.0, 1.0]`. Identical vectors yield `1.0`.
/// Returns `0.0` if either vector has zero norm to avoid division by zero.
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "cosine_similarity: vectors must have the same length"
    );
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Compute the Euclidean distance between two vectors.
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "euclidean_distance: vectors must have the same length"
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Return a new normalized (L2 unit) vector.
///
/// Returns a zero-filled vector of the same length if the input norm is zero.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm == 0.0 {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| x / norm).collect()
}

/// Trait for clients that can generate embeddings from text.
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Embed a batch of texts into vectors.
    ///
    /// The returned vector has the same length and order as the input.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>, KairoError>;
}

/// A local embedding client using deterministic vector generation.
///
/// Generates pseudo-random embeddings from input text hashes.
/// Suitable for development and testing without ML framework dependencies.
/// For production use, switch to `RemoteEmbeddingClient` with a hosted
/// embedding API (OpenAI, Cohere, etc.).
#[derive(Debug, Clone)]
pub struct LocalEmbeddingClient {
    dimension: usize,
    seed: u64,
}

impl LocalEmbeddingClient {
    /// Create a new local client with the target embedding dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            seed: 42,
        }
    }

    /// Set a custom seed for deterministic generation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generate a deterministic pseudo-random vector from text.
    fn generate_vector(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        text.hash(&mut hasher);
        let state = hasher.finish();

        let mut vec = Vec::with_capacity(self.dimension);
        let mut s = state;
        for _ in 0..self.dimension {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let normalized = (s as f32) / (u64::MAX as f32);
            vec.push(normalized * 2.0 - 1.0);
        }
        normalize(&vec)
    }
}

#[async_trait]
impl EmbeddingClient for LocalEmbeddingClient {
    #[instrument(skip(self, texts), fields(count = texts.len()))]
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>, KairoError> {
        debug!("Generating local embeddings for {} texts", texts.len());
        let embeddings: Vec<Embedding> = texts
            .into_iter()
            .map(|text| {
                let vector = self.generate_vector(&text);
                Embedding::new(vector).with_metadata(EmbeddingMetadata {
                    text: text.clone(),
                    model: "local-deterministic".to_string(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                })
            })
            .collect();
        Ok(embeddings)
    }
}

/// Configuration for a remote (OpenAI-style) embedding API.
#[derive(Debug, Clone)]
pub struct RemoteEmbeddingConfig {
    /// Base URL of the API (e.g. `https://api.openai.com/v1`).
    pub base_url: String,
    /// API key for authentication.
    pub api_key: String,
    /// Model identifier (e.g. `text-embedding-3-small`).
    pub model: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Max retries on transient failures.
    pub max_retries: u32,
}

impl Default for RemoteEmbeddingConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            model: "text-embedding-3-small".to_string(),
            timeout_secs: 60,
            max_retries: 3,
        }
    }
}

/// A remote embedding client that calls an OpenAI-compatible embeddings endpoint.
#[derive(Debug, Clone)]
pub struct RemoteEmbeddingClient {
    config: RemoteEmbeddingConfig,
    client: reqwest::Client,
}

/// OpenAI-compatible embedding request body.
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

/// OpenAI-compatible embedding response.
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDataItem>,
    model: String,
    usage: Option<EmbeddingUsage>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingDataItem {
    embedding: Vec<f32>,
    index: usize,
    object: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

impl RemoteEmbeddingClient {
    /// Create a new remote client from configuration.
    pub fn new(config: RemoteEmbeddingConfig) -> Result<Self, KairoError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| KairoError::Internal(format!("Failed to build HTTP client: {e}")))?;
        Ok(Self { config, client })
    }

    /// Send a single embedding request with retries.
    async fn request_embeddings(
        &self,
        texts: Vec<String>,
    ) -> Result<Vec<Embedding>, KairoError> {
        let url = format!("{}/embeddings", self.config.base_url.trim_end_matches('/'));

        let body = EmbeddingRequest {
            model: self.config.model.clone(),
            input: texts.clone(),
        };

        let mut last_error = None;
        for attempt in 0..self.config.max_retries {
            debug!("Embedding request attempt {}/{} to {}", attempt + 1, self.config.max_retries, url);

            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.config.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let payload: EmbeddingResponse = resp.json().await.map_err(|e| {
                            KairoError::Internal(format!("Failed to parse embedding response: {e}"))
                        })?;
                        let embeddings = payload
                            .data
                            .into_iter()
                            .map(|item| {
                                Embedding::new(item.embedding).with_metadata(EmbeddingMetadata {
                                    text: texts.get(item.index).cloned().unwrap_or_default(),
                                    model: payload.model.clone(),
                                    created_at: chrono::Utc::now().to_rfc3339(),
                                })
                            })
                            .collect();
                        return Ok(embeddings);
                    } else {
                        let text = resp
                            .text()
                            .await
                            .unwrap_or_else(|_| "<failed to read body>".to_string());
                        warn!("Embedding API error (status {}): {}", status, text);
                        last_error = Some(KairoError::Internal(format!(
                            "Embedding API returned {status}: {text}"
                        )));
                        if status.is_server_error() {
                            tokio::time::sleep(std::time::Duration::from_millis(500 * (attempt + 1) as u64)).await;
                            continue;
                        } else {
                            return Err(last_error.unwrap());
                        }
                    }
                }
                Err(e) if e.is_timeout() || e.is_connect() => {
                    warn!("Embedding request failed (transient): {e}");
                    last_error = Some(KairoError::Internal(format!("Embedding request failed: {e}")));
                    tokio::time::sleep(std::time::Duration::from_millis(500 * (attempt + 1) as u64)).await;
                }
                Err(e) => {
                    return Err(KairoError::Internal(format!("Embedding request failed: {e}")));
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            KairoError::Internal("Embedding request exhausted all retries".to_string())
        }))
    }
}

#[async_trait]
impl EmbeddingClient for RemoteEmbeddingClient {
    #[instrument(skip(self, texts), fields(count = texts.len()))]
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>, KairoError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        debug!("Requesting remote embeddings for {} texts", texts.len());
        self.request_embeddings(texts).await
    }
}

/// Trait for stores that persist and retrieve embeddings.
#[async_trait]
pub trait EmbeddingStore: Send + Sync {
    /// Insert or update an embedding.
    async fn upsert(&self, embedding: Embedding) -> Result<(), KairoError>;

    /// Retrieve an embedding by id.
    async fn get(&self, id: &str) -> Result<Option<Embedding>, KairoError>;

    /// Delete an embedding by id.
    async fn delete(&self, id: &str) -> Result<bool, KairoError>;

    /// Find the `k` embeddings most similar to the query vector.
    ///
    /// Results are ordered from most similar to least similar.
    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<(Embedding, f32)>, KairoError>;

    /// Return the total number of stored embeddings.
    async fn len(&self) -> Result<usize, KairoError>;

    /// Return true if the store contains no embeddings.
    async fn is_empty(&self) -> Result<bool, KairoError> {
        Ok(self.len().await? == 0)
    }
}

use std::sync::Arc;
use tokio::sync::RwLock;

/// An in-memory embedding store backed by a `HashMap`.
///
/// Search is performed via brute-force cosine similarity over all stored vectors.
/// This is suitable for small datasets and prototyping.
#[derive(Debug, Clone, Default)]
pub struct InMemoryEmbeddingStore {
    data: Arc<RwLock<HashMap<String, Embedding>>>,
}

impl InMemoryEmbeddingStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl EmbeddingStore for InMemoryEmbeddingStore {
    async fn upsert(&self, embedding: Embedding) -> Result<(), KairoError> {
        let mut guard = self.data.write().await;
        guard.insert(embedding.id.clone(), embedding);
        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<Embedding>, KairoError> {
        let guard = self.data.read().await;
        Ok(guard.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<bool, KairoError> {
        let mut guard = self.data.write().await;
        Ok(guard.remove(id).is_some())
    }

    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<(Embedding, f32)>, KairoError> {
        let guard = self.data.read().await;
        let mut scored: Vec<(Embedding, f32)> = guard
            .values()
            .map(|emb| {
                let score = cosine_similarity(query, &emb.vector);
                (emb.clone(), score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    async fn len(&self) -> Result<usize, KairoError> {
        let guard = self.data.read().await;
        Ok(guard.len())
    }
}

pub mod prelude {
    //! Convenient imports for consumers of this crate.
    pub use super::{
        cosine_similarity, dot_product, euclidean_distance, normalize, Embedding,
        EmbeddingClient, EmbeddingMetadata, EmbeddingStore, InMemoryEmbeddingStore,
        LocalEmbeddingClient, RemoteEmbeddingClient, RemoteEmbeddingConfig,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b), 32.0);
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        assert_eq!(l2_norm(&v), 5.0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0];
        let b = vec![4.0, 6.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = normalize(&v);
        let norm = l2_norm(&normalized);
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let normalized = normalize(&v);
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_embedding_creation() {
        let emb = Embedding::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.dim(), 3);
        assert!(!emb.id.is_empty());
    }

    #[test]
    fn test_embedding_with_id() {
        let emb = Embedding::with_id("test-123", vec![1.0, 2.0]);
        assert_eq!(emb.id, "test-123");
    }

    #[tokio::test]
    async fn test_in_memory_embedding_store() {
        let store = InMemoryEmbeddingStore::new();
        let emb = Embedding::with_id("test-1", vec![1.0, 0.0, 0.0]);
        store.upsert(emb.clone()).await.unwrap();

        assert_eq!(store.len().await.unwrap(), 1);
        assert!(!store.is_empty().await.unwrap());

        let retrieved = store.get("test-1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test-1");

        let results = store.search(&[1.0, 0.0, 0.0], 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].1 - 1.0).abs() < 1e-6);

        let deleted = store.delete("test-1").await.unwrap();
        assert!(deleted);
        assert_eq!(store.len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_local_embedding_client() {
        let client = LocalEmbeddingClient::new(128);
        let embeddings = client.embed(vec!["hello".into(), "world".into()]).await.unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].dim(), 128);
        assert!(!embeddings[0].id.is_empty());
    }

    #[tokio::test]
    async fn test_local_embedding_client_deterministic() {
        let client1 = LocalEmbeddingClient::new(64).with_seed(42);
        let client2 = LocalEmbeddingClient::new(64).with_seed(42);

        let emb1 = client1.embed(vec!["test".into()]).await.unwrap();
        let emb2 = client2.embed(vec!["test".into()]).await.unwrap();

        assert_eq!(emb1[0].vector, emb2[0].vector);
    }
}
