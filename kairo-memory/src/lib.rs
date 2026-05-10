use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, instrument};
use uuid::Uuid;

use kairo_core::{KairoError, Message, Role};
use kairo_embeddings::{cosine_similarity, normalize, Embedding, EmbeddingStore, InMemoryEmbeddingStore};

/// A memory entry representing a single interaction or observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub content: String,
    pub embedding: Option<Embedding>,
    pub metadata: MemoryMetadata,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryMetadata {
    pub agent_id: Option<Uuid>,
    pub session_id: Option<Uuid>,
    pub entry_type: MemoryType,
    pub importance: f32,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryType {
    Episodic,
    Semantic,
    Procedural,
    Custom(String),
}

impl Default for MemoryType {
    fn default() -> Self {
        MemoryType::Episodic
    }
}

/// Episodic memory stores the agent's personal experiences and interactions.
pub struct EpisodicMemory {
    entries: RwLock<Vec<MemoryEntry>>,
}

impl EpisodicMemory {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }

    #[instrument(skip(self, entry))]
    pub async fn store(&self, entry: MemoryEntry) {
        debug!(entry_id = %entry.id, "storing episodic memory");
        let mut entries = self.entries.write().await;
        entries.push(entry);
    }

    pub async fn retrieve_recent(&self, limit: usize) -> Vec<MemoryEntry> {
        let entries = self.entries.read().await;
        entries.iter().rev().take(limit).cloned().collect()
    }

    pub async fn retrieve_by_session(&self, session_id: Uuid) -> Vec<MemoryEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| e.metadata.session_id == Some(session_id))
            .cloned()
            .collect()
    }
}

impl Default for EpisodicMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Semantic memory stores factual knowledge via embeddings.
pub struct SemanticMemory {
    store: InMemoryEmbeddingStore,
}

impl SemanticMemory {
    pub fn new() -> Self {
        Self {
            store: InMemoryEmbeddingStore::new(),
        }
    }

    pub async fn store(&self, entry: MemoryEntry) -> Result<(), KairoError> {
        if let Some(embedding) = &entry.embedding {
            self.store.upsert(embedding.clone()).await?;
        }
        Ok(())
    }

    pub async fn search(&self, query_embedding: &Embedding, top_k: usize) -> Result<Vec<(Embedding, f32)>, KairoError> {
        self.store.search(&query_embedding.vector, top_k).await
    }
}

impl Default for SemanticMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid memory combines episodic and semantic search.
pub struct HybridMemory {
    episodic: Arc<EpisodicMemory>,
    semantic: Arc<SemanticMemory>,
    entries_by_id: RwLock<HashMap<String, MemoryEntry>>,
}

impl HybridMemory {
    pub fn new() -> Self {
        Self {
            episodic: Arc::new(EpisodicMemory::new()),
            semantic: Arc::new(SemanticMemory::new()),
            entries_by_id: RwLock::new(HashMap::new()),
        }
    }

    pub async fn store(&self, entry: MemoryEntry) -> Result<(), KairoError> {
        let id = entry.id.to_string();
        self.episodic.store(entry.clone()).await;
        self.semantic.store(entry.clone()).await?;
        let mut map = self.entries_by_id.write().await;
        map.insert(id, entry);
        Ok(())
    }

    pub async fn search(
        &self,
        query_embedding: &Embedding,
        top_k: usize,
    ) -> Result<Vec<MemoryEntry>, KairoError> {
        let semantic_results = self.semantic.search(query_embedding, top_k).await?;
        let map = self.entries_by_id.read().await;
        let mut results = Vec::new();
        for (embedding, _score) in semantic_results {
            if let Some(entry) = map.get(&embedding.id) {
                results.push(entry.clone());
            }
        }
        Ok(results)
    }

    pub async fn recent(&self, limit: usize) -> Vec<MemoryEntry> {
        self.episodic.retrieve_recent(limit).await
    }

    pub async fn to_messages(&self, limit: usize) -> Vec<Message> {
        let entries = self.episodic.retrieve_recent(limit).await;
        entries
            .into_iter()
            .map(|e| Message {
                role: Role::User,
                content: e.content,
                name: None,
                timestamp: e.created_at,
            })
            .collect()
    }
}

impl Default for HybridMemory {
    fn default() -> Self {
        Self::new()
    }
}
