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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use kairo_tools::{CalculatorTool, ToolRegistry};

    use super::*;

    #[tokio::test]
    async fn test_tool_dispatcher_routes_to_registered_tool() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(CalculatorTool)).await;
        let dispatcher = ToolDispatcher::new(registry);
        let call = ToolCall {
            tool_name: "calculator".into(),
            arguments: serde_json::json!({"expression": "1+1"}),
        };
        let output = dispatcher.dispatch(&call).await.unwrap();
        assert!(output.success);
        assert_eq!(
            output.result.get("result").unwrap().as_f64().unwrap(),
            2.0
        );
    }
}
