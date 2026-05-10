use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::model::ModelId;
use crate::task::TaskType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: Uuid,
    pub task_type: TaskType,
    pub description: String,
    pub input: serde_json::Value,
    pub expected_output: Option<String>,
    pub assigned_model: Option<ModelId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtask {
    pub id: Uuid,
    pub parent_id: Uuid,
    pub task_type: TaskType,
    pub description: String,
    pub dependencies: Vec<Uuid>,
    pub status: TaskStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WorkflowStatus {
    Draft,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub id: Uuid,
    pub name: String,
    pub tasks: Vec<Task>,
    pub subtasks: Vec<Subtask>,
    pub status: WorkflowStatus,
}
