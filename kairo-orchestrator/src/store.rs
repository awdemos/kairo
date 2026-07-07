use async_trait::async_trait;
use kairo_core::error::KairoError;
use kairo_core::{TaskStatus, Workflow, WorkflowStatus};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct TaskState {
    pub status: TaskStatus,
    pub output: Option<String>,
}

impl Default for TaskState {
    fn default() -> Self {
        Self {
            status: TaskStatus::Pending,
            output: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkflowRecord {
    pub workflow: Workflow,
    pub statuses: HashMap<Uuid, TaskState>,
    pub overall_status: WorkflowStatus,
}

#[async_trait]
pub trait WorkflowStore: Send + Sync {
    async fn create(&self, workflow: Workflow) -> Result<(), KairoError>;
    async fn get(&self, id: Uuid) -> Result<Option<WorkflowRecord>, KairoError>;
    async fn update_task(
        &self,
        workflow_id: Uuid,
        task_id: Uuid,
        state: TaskState,
    ) -> Result<(), KairoError>;
    async fn set_status(&self, workflow_id: Uuid, status: WorkflowStatus) -> Result<(), KairoError>;
}

pub struct InMemoryWorkflowStore {
    records: Arc<RwLock<HashMap<Uuid, WorkflowRecord>>>,
}

impl InMemoryWorkflowStore {
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryWorkflowStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for InMemoryWorkflowStore {
    fn clone(&self) -> Self {
        Self {
            records: Arc::clone(&self.records),
        }
    }
}

#[async_trait]
impl WorkflowStore for InMemoryWorkflowStore {
    async fn create(&self, workflow: Workflow) -> Result<(), KairoError> {
        let mut records = self.records.write().await;
        let mut statuses = HashMap::new();
        for task in &workflow.tasks {
            statuses.insert(task.id, TaskState::default());
        }
        records.insert(
            workflow.id,
            WorkflowRecord {
                workflow,
                statuses,
                overall_status: WorkflowStatus::Draft,
            },
        );
        Ok(())
    }

    async fn get(&self, id: Uuid) -> Result<Option<WorkflowRecord>, KairoError> {
        let records = self.records.read().await;
        Ok(records.get(&id).cloned())
    }

    async fn update_task(
        &self,
        workflow_id: Uuid,
        task_id: Uuid,
        state: TaskState,
    ) -> Result<(), KairoError> {
        let mut records = self.records.write().await;
        let record = records
            .get_mut(&workflow_id)
            .ok_or_else(|| KairoError::Workflow(format!("Workflow {} not found", workflow_id)))?;
        record.statuses.insert(task_id, state);
        Ok(())
    }

    async fn set_status(&self, workflow_id: Uuid, status: WorkflowStatus) -> Result<(), KairoError> {
        let mut records = self.records.write().await;
        let record = records
            .get_mut(&workflow_id)
            .ok_or_else(|| KairoError::Workflow(format!("Workflow {} not found", workflow_id)))?;
        record.overall_status = status;
        Ok(())
    }
}
