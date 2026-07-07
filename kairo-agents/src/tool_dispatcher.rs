use kairo_core::{KairoError, ToolInput, ToolOutput};
use kairo_tools::ToolRegistry;

use crate::ToolCall;

pub struct ToolDispatcher {
    registry: ToolRegistry,
}

impl ToolDispatcher {
    pub fn new(registry: ToolRegistry) -> Self {
        Self { registry }
    }

    pub async fn dispatch(&self, call: &ToolCall) -> Result<ToolOutput, KairoError> {
        let input = ToolInput {
            arguments: call.arguments.clone(),
        };
        self.registry.execute(&call.tool_name, input).await
    }
}
