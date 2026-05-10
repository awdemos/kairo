use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TaskType {
    Research,
    CodeGeneration,
    CodeReview,
    MediaGeneration,
    DataAnalysis,
    Summarization,
    Translation,
    Reasoning,
    CreativeWriting,
    ToolUse,
    MultiStep,
    Custom(String),
}
