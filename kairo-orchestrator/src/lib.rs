use std::collections::HashMap;
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

use kairo_core::{Context, KairoError, Workflow, WorkflowStatus};

pub mod dag;
pub mod executor;
pub mod runner;
pub mod store;

pub use runner::TaskRunner;
pub use store::{InMemoryWorkflowStore, TaskState, WorkflowRecord, WorkflowStore};

use crate::dag::DagBuilder;
use crate::executor::Executor;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkflowResult {
    pub workflow_id: Uuid,
    pub outputs: HashMap<Uuid, String>,
    pub status: WorkflowStatus,
}

pub struct WorkflowEngine {
    store: InMemoryWorkflowStore,
}

impl WorkflowEngine {
    pub fn new() -> Self {
        Self {
            store: InMemoryWorkflowStore::new(),
        }
    }

    #[instrument(skip(self))]
    pub async fn register(&self, workflow: Workflow) -> Result<(), KairoError> {
        self.store.create(workflow).await
    }

    #[instrument(skip(self, ctx, runner))]
    pub async fn execute(
        &self,
        workflow_id: Uuid,
        ctx: Context,
        runner: Arc<dyn TaskRunner>,
    ) -> Result<WorkflowResult, KairoError> {
        let record = self
            .store
            .get(workflow_id)
            .await?
            .ok_or_else(|| KairoError::Workflow(format!("Workflow {} not found", workflow_id)))?;
        let dag = DagBuilder::build(&record.workflow)?;
        let executor = Executor::new(self.store.clone(), runner);
        executor.execute(workflow_id, dag, ctx).await
    }

    pub async fn get_status(&self, workflow_id: Uuid) -> Option<WorkflowStatus> {
        self.store
            .get(workflow_id)
            .await
            .ok()
            .flatten()
            .map(|r| r.overall_status)
    }

    pub async fn get_outputs(&self, workflow_id: Uuid) -> Option<HashMap<Uuid, String>> {
        self.store
            .get(workflow_id)
            .await
            .ok()
            .flatten()
            .map(|r| {
                let mut outputs = HashMap::new();
                for (id, state) in r.statuses {
                    if let Some(output) = state.output {
                        outputs.insert(id, output);
                    }
                }
                outputs
            })
    }
}

impl Default for WorkflowEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairo_core::{Workflow, WorkflowStatus};

    #[tokio::test]
    async fn test_workflow_engine_register() {
        let engine = WorkflowEngine::new();
        let workflow = Workflow {
            id: Uuid::new_v4(),
            name: "test".into(),
            tasks: vec![],
            subtasks: vec![],
            status: WorkflowStatus::Draft,
        };

        let result = engine.register(workflow).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_workflow_engine_get_status() {
        let engine = WorkflowEngine::new();
        let id = Uuid::new_v4();
        let workflow = Workflow {
            id,
            name: "test".into(),
            tasks: vec![],
            subtasks: vec![],
            status: WorkflowStatus::Draft,
        };

        engine.register(workflow).await.unwrap();
        let status = engine.get_status(id).await;
        assert_eq!(status, Some(WorkflowStatus::Draft));
    }

    #[tokio::test]
    async fn test_workflow_engine_missing_workflow() {
        let engine = WorkflowEngine::new();
        let id = Uuid::new_v4();
        let status = engine.get_status(id).await;
        assert!(status.is_none());
    }

    #[test]
    fn test_workflow_result_creation() {
        let id = Uuid::new_v4();
        let result = WorkflowResult {
            workflow_id: id,
            outputs: HashMap::new(),
            status: WorkflowStatus::Completed,
        };
        assert_eq!(result.workflow_id, id);
        assert_eq!(result.status, WorkflowStatus::Completed);
    }
}
