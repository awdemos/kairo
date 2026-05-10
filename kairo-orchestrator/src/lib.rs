use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument};
use uuid::Uuid;

use kairo_core::{
    Agent, CompletionOptions, CompletionResponse, Context, KairoError, Message, ModelId, Role,
    Task, TaskStatus, Workflow, WorkflowStatus,
};
use kairo_agents::ReActAgent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub workflow_id: Uuid,
    pub outputs: HashMap<Uuid, String>,
    pub status: WorkflowStatus,
}

pub struct WorkflowEngine {
    workflows: RwLock<HashMap<Uuid, WorkflowState>>,
}

#[derive(Debug)]
struct WorkflowState {
    workflow: Workflow,
    graph: DiGraph<WorkflowNode, ()>,
    node_map: HashMap<Uuid, NodeIndex>,
    outputs: HashMap<Uuid, String>,
}

#[derive(Debug, Clone)]
struct WorkflowNode {
    id: Uuid,
    name: String,
    status: TaskStatus,
    agent: Option<Arc<ReActAgent>>,
}

impl WorkflowEngine {
    pub fn new() -> Self {
        Self {
            workflows: RwLock::new(HashMap::new()),
        }
    }

    #[instrument(skip(self))]
    pub async fn register(&self, workflow: Workflow, agents: HashMap<Uuid, Arc<ReActAgent>>) -> Result<(), KairoError> {
        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();

        for task in &workflow.tasks {
            let node = WorkflowNode {
                id: task.id,
                name: task.description.clone(),
                status: TaskStatus::Pending,
                agent: agents.get(&task.id).cloned(),
            };
            let idx = graph.add_node(node);
            node_map.insert(task.id, idx);
        }

        for subtask in &workflow.subtasks {
            if let Some(&from) = node_map.get(&subtask.parent_id) {
                for dep_id in &subtask.dependencies {
                    if let Some(&to) = node_map.get(dep_id) {
                        graph.add_edge(to, from, ());
                    }
                }
            }
        }

        let state = WorkflowState {
            workflow,
            graph,
            node_map,
            outputs: HashMap::new(),
        };

        let mut workflows = self.workflows.write().await;
        workflows.insert(state.workflow.id, state);
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn execute(&self, workflow_id: Uuid, ctx: Context) -> Result<WorkflowResult, KairoError> {
        let mut workflows = self.workflows.write().await;
        let state = workflows.get_mut(&workflow_id)
            .ok_or_else(|| KairoError::Workflow(format!("Workflow {} not found", workflow_id)))?;

        let mut topo = Topo::new(&state.graph);
        let mut execution_order = Vec::new();

        while let Some(node_idx) = topo.next(&state.graph) {
            execution_order.push(node_idx);
        }

        for node_idx in execution_order {
            let node = &mut state.graph[node_idx];
            node.status = TaskStatus::InProgress;
            debug!(task_id = %node.id, name = %node.name, "executing workflow task");

            if let Some(agent) = &node.agent {
                let mut task_ctx = ctx.clone();
                task_ctx.workflow_id = Some(workflow_id);

                let result = agent.run(task_ctx).await;
                match result {
                    Ok(agent_result) => {
                        state.outputs.insert(node.id, agent_result.output.clone());
                        node.status = TaskStatus::Completed;
                        info!(task_id = %node.id, "task completed");
                    }
                    Err(e) => {
                        node.status = TaskStatus::Failed;
                        error!(task_id = %node.id, error = %e, "task failed");
                        state.workflow.status = WorkflowStatus::Failed;
                        return Ok(WorkflowResult {
                            workflow_id,
                            outputs: state.outputs.clone(),
                            status: WorkflowStatus::Failed,
                        });
                    }
                }
            } else {
                node.status = TaskStatus::Completed;
                state.outputs.insert(node.id, "No agent assigned".into());
            }
        }

        state.workflow.status = WorkflowStatus::Completed;
        Ok(WorkflowResult {
            workflow_id,
            outputs: state.outputs.clone(),
            status: WorkflowStatus::Completed,
        })
    }

    pub async fn get_status(&self, workflow_id: Uuid) -> Option<WorkflowStatus> {
        let workflows = self.workflows.read().await;
        workflows.get(&workflow_id).map(|s| s.workflow.status.clone())
    }

    pub async fn get_outputs(&self, workflow_id: Uuid) -> Option<HashMap<Uuid, String>> {
        let workflows = self.workflows.read().await;
        workflows.get(&workflow_id).map(|s| s.outputs.clone())
    }
}

impl Default for WorkflowEngine {
    fn default() -> Self {
        Self::new()
    }
}
