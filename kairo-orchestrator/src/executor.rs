use kairo_core::context::Context;
use kairo_core::error::KairoError;
use kairo_core::WorkflowStatus;
use kairo_core::TaskStatus;
use petgraph::algo::toposort;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::dag::WorkflowDag;
use crate::runner::TaskRunner;
use crate::store::{TaskState, WorkflowStore};
use crate::WorkflowResult;

pub struct Executor<S: WorkflowStore> {
    store: S,
    runner: Arc<dyn TaskRunner>,
}

impl<S: WorkflowStore> Executor<S> {
    pub fn new(store: S, runner: Arc<dyn TaskRunner>) -> Self {
        Self { store, runner }
    }

    pub async fn execute(
        &self,
        workflow_id: Uuid,
        dag: WorkflowDag,
        ctx: Context,
    ) -> Result<WorkflowResult, KairoError> {
        let order = toposort(&dag.graph, None)
            .map_err(|_| KairoError::Workflow("Workflow contains a dependency cycle".into()))?;

        for node_idx in order {
            let task_id = dag.graph[node_idx];
            debug!(task_id = %task_id, "executing workflow task");

            self.store
                .update_task(
                    workflow_id,
                    task_id,
                    TaskState {
                        status: TaskStatus::InProgress,
                        output: None,
                    },
                )
                .await?;

            let mut task_ctx = ctx.clone();
            task_ctx.workflow_id = Some(workflow_id);

            match self.runner.run(task_ctx).await {
                Ok(result) => {
                    self.store
                        .update_task(
                            workflow_id,
                            task_id,
                            TaskState {
                                status: TaskStatus::Completed,
                                output: Some(result.output.clone()),
                            },
                        )
                        .await?;
                    info!(task_id = %task_id, "task completed");
                }
                Err(e) => {
                    self.store
                        .update_task(
                            workflow_id,
                            task_id,
                            TaskState {
                                status: TaskStatus::Failed,
                                output: None,
                            },
                        )
                        .await?;
                    self.store.set_status(workflow_id, WorkflowStatus::Failed).await?;
                    error!(task_id = %task_id, error = %e, "task failed");
                    return Err(KairoError::Workflow(format!("Task {} failed: {}", task_id, e)));
                }
            }
        }

        self.store
            .set_status(workflow_id, WorkflowStatus::Completed)
            .await?;

        let record = self.store.get(workflow_id).await?;
        let mut outputs = HashMap::new();
        if let Some(record) = record {
            for (task_id, state) in &record.statuses {
                if let Some(output) = &state.output {
                    outputs.insert(*task_id, output.clone());
                }
            }
        }

        Ok(WorkflowResult {
            workflow_id,
            outputs,
            status: WorkflowStatus::Completed,
        })
    }
}
