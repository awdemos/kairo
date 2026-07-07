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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DagBuilder;
    use crate::InMemoryWorkflowStore;
    use kairo_agents::AgentResult;
    use kairo_core::{Context, KairoError, Task, TaskType, TokenUsage, Workflow, WorkflowStatus};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use uuid::Uuid;

    struct FakeRunner {
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl TaskRunner for FakeRunner {
        async fn run(&self, _ctx: Context) -> Result<AgentResult, KairoError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(AgentResult {
                output: "done".into(),
                thoughts: vec![],
                tool_calls: vec![],
                token_usage: TokenUsage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                },
            })
        }
    }

    #[tokio::test]
    async fn test_executor_runs_single_task() {
        let workflow = Workflow {
            id: Uuid::new_v4(),
            name: "test".into(),
            tasks: vec![Task {
                id: Uuid::new_v4(),
                task_type: TaskType::Research,
                description: "task 1".into(),
                input: serde_json::Value::Null,
                expected_output: None,
                assigned_model: None,
            }],
            subtasks: vec![],
            status: WorkflowStatus::Draft,
        };
        let store = InMemoryWorkflowStore::new();
        store.create(workflow.clone()).await.unwrap();
        let dag = DagBuilder::build(&workflow).unwrap();
        let runner = Arc::new(FakeRunner { calls: AtomicUsize::new(0) });
        let executor = Executor::new(store.clone(), runner.clone());
        let result = executor
            .execute(workflow.id, dag, Context::new(Uuid::new_v4()))
            .await
            .unwrap();
        assert_eq!(result.status, WorkflowStatus::Completed);
        assert_eq!(runner.calls.load(Ordering::SeqCst), 1);
    }
}
