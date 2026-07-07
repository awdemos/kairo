use kairo_core::Workflow;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct WorkflowDag {
    pub graph: DiGraph<Uuid, ()>,
    pub node_map: HashMap<Uuid, NodeIndex>,
}

impl WorkflowDag {
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }
}

pub struct DagBuilder;

impl DagBuilder {
    pub fn build(workflow: &Workflow) -> Result<WorkflowDag, kairo_core::error::KairoError> {
        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();

        for task in &workflow.tasks {
            let idx = graph.add_node(task.id);
            node_map.insert(task.id, idx);
        }

        for subtask in &workflow.subtasks {
            let &from = node_map
                .get(&subtask.parent_id)
                .ok_or_else(|| kairo_core::error::KairoError::Workflow(format!("Unknown parent task: {}", subtask.parent_id)))?;
            for dep_id in &subtask.dependencies {
                let &to = node_map
                    .get(dep_id)
                    .ok_or_else(|| kairo_core::error::KairoError::Workflow(format!("Unknown dependency: {}", dep_id)))?;
                graph.add_edge(to, from, ());
            }
        }

        Ok(WorkflowDag { graph, node_map })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairo_core::{Subtask, Task, TaskStatus, TaskType, Workflow, WorkflowStatus};
    use uuid::Uuid;

    #[test]
    fn builds_dag_for_empty_workflow() {
        let workflow = Workflow {
            id: Uuid::new_v4(),
            name: "test".into(),
            tasks: vec![],
            subtasks: vec![],
            status: WorkflowStatus::Draft,
        };
        let dag = DagBuilder::build(&workflow).unwrap();
        assert_eq!(dag.node_count(), 0);
    }

    #[test]
    fn detects_unknown_dependency() {
        let task_id = Uuid::new_v4();
        let workflow = Workflow {
            id: Uuid::new_v4(),
            name: "test".into(),
            tasks: vec![Task {
                id: task_id,
                task_type: TaskType::Research,
                description: "a".into(),
                input: serde_json::Value::Null,
                expected_output: None,
                assigned_model: None,
            }],
            subtasks: vec![Subtask {
                id: Uuid::new_v4(),
                parent_id: task_id,
                task_type: TaskType::Research,
                description: "b".into(),
                dependencies: vec![Uuid::new_v4()],
                status: TaskStatus::Pending,
            }],
            status: WorkflowStatus::Draft,
        };
        assert!(DagBuilder::build(&workflow).is_err());
    }
}
