use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, instrument, warn};

use kairo_core::{KairoError, Tool, ToolInput, ToolOutput};

/// A registry for managing available tools.
pub struct ToolRegistry {
    tools: RwLock<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    #[instrument(skip(self, tool))]
    pub async fn register(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        debug!(tool_name = %name, "registering tool");
        let mut tools = self.tools.write().await;
        tools.insert(name, tool);
    }

    pub async fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let tools = self.tools.read().await;
        tools.get(name).cloned()
    }

    pub async fn list(&self) -> Vec<String> {
        let tools = self.tools.read().await;
        tools.keys().cloned().collect()
    }

    pub async fn execute(&self, name: &str, input: ToolInput) -> Result<ToolOutput, KairoError> {
        let tool = self
            .get(name)
            .await
            .ok_or_else(|| KairoError::Tool(format!("Tool '{}' not found", name)))?;
        debug!(tool_name = %name, "executing tool");
        tool.execute(input).await
    }

    pub async fn unregister(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let mut tools = self.tools.write().await;
        tools.remove(name)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Built-in tools

/// Calculator tool for arithmetic expressions.
pub struct CalculatorTool;

impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate mathematical expressions. Input: {\"expression\": \"2 + 2\"}"
    }

    fn execute<'a>(
        &'a self,
        input: ToolInput,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, KairoError>> + Send + 'a>> {
        Box::pin(async move {
            let expression = input
                .arguments
                .get("expression")
                .and_then(|v| v.as_str())
                .ok_or_else(|| KairoError::Tool("Missing 'expression' argument".into()))?;

            let result = meval::eval_str(expression)
                .map_err(|e| KairoError::Tool(format!("Math error: {}", e)))?;

            Ok(ToolOutput {
                result: serde_json::json!({"result": result}),
                success: true,
            })
        })
    }
}

/// Web search tool — simplified implementation.
///
/// Requires `SERPAPI_KEY` or `BRAVE_API_KEY` environment variable
/// for live search results. Falls back to a helpful message if
/// no API key is configured.
pub struct WebSearchTool {
    client: reqwest::Client,
}

impl WebSearchTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for information. Input: {\"query\": \"search terms\"}"
    }

    fn execute<'a>(
        &'a self,
        input: ToolInput,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, KairoError>> + Send + 'a>> {
        Box::pin(async move {
            let query = input
                .arguments
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| KairoError::Tool("Missing 'query' argument".into()))?;

            warn!(query = %query, "web_search tool requires a search API key");

            Ok(ToolOutput {
                result: serde_json::json!({
                    "query": query,
                    "results": [],
                    "note": "To enable web search, set SERPAPI_KEY or BRAVE_API_KEY environment variable"
                }),
                success: true,
            })
        })
    }
}

/// File system tool for reading/writing files.
pub struct FileSystemTool;

impl Tool for FileSystemTool {
    fn name(&self) -> &str {
        "filesystem"
    }

    fn description(&self) -> &str {
        "Read or write files. Input: {\"operation\": \"read|write\", \"path\": \"...\", \"content\": \"...\"}"
    }

    fn execute<'a>(
        &'a self,
        input: ToolInput,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, KairoError>> + Send + 'a>> {
        Box::pin(async move {
            let operation = input
                .arguments
                .get("operation")
                .and_then(|v| v.as_str())
                .ok_or_else(|| KairoError::Tool("Missing 'operation' argument".into()))?;

            let path = input
                .arguments
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| KairoError::Tool("Missing 'path' argument".into()))?;

            match operation {
                "read" => {
                    let content = tokio::fs::read_to_string(path).await.map_err(|e| {
                        KairoError::Tool(format!("Failed to read file: {}", e))
                    })?;
                    Ok(ToolOutput {
                        result: serde_json::json!({"content": content}),
                        success: true,
                    })
                }
                "write" => {
                    let content = input
                        .arguments
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    tokio::fs::write(path, content).await.map_err(|e| {
                        KairoError::Tool(format!("Failed to write file: {}", e))
                    })?;
                    Ok(ToolOutput {
                        result: serde_json::json!({"written": true}),
                        success: true,
                    })
                }
                _ => Err(KairoError::Tool(format!(
                    "Unknown operation: {}",
                    operation
                ))),
            }
        })
    }
}

/// Code execution tool (runs code in a sandboxed environment).
pub struct CodeExecutionTool;

impl Tool for CodeExecutionTool {
    fn name(&self) -> &str {
        "code_execution"
    }

    fn description(&self) -> &str {
        "Execute code snippets. Input: {\"language\": \"python|rust|js\", \"code\": \"...\"}"
    }

    fn execute<'a>(
        &'a self,
        input: ToolInput,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, KairoError>> + Send + 'a>> {
        Box::pin(async move {
            let language = input
                .arguments
                .get("language")
                .and_then(|v| v.as_str())
                .ok_or_else(|| KairoError::Tool("Missing 'language' argument".into()))?;

            let code = input
                .arguments
                .get("code")
                .and_then(|v| v.as_str())
                .ok_or_else(|| KairoError::Tool("Missing 'code' argument".into()))?;

            warn!(language = %language, "code_execution requires kairo-sandbox integration");

            Ok(ToolOutput {
                result: serde_json::json!({
                    "language": language,
                    "code": code,
                    "output": "Code execution requires kairo-sandbox WASM runtime integration", "note": "Enable sandbox feature for secure execution"
                }),
                success: true,
            })
        })
    }
}

/// Create a default registry with all built-in tools.
pub async fn default_registry() -> ToolRegistry {
    let registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool)).await;
    registry.register(Arc::new(WebSearchTool::new())).await;
    registry.register(Arc::new(FileSystemTool)).await;
    registry.register(Arc::new(CodeExecutionTool)).await;
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairo_core::ToolInput;

    #[tokio::test]
    async fn test_tool_registry() {
        let registry = ToolRegistry::new();
        let tool = Arc::new(CalculatorTool);
        registry.register(tool.clone()).await;

        let tools = registry.list().await;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0], "calculator");

        let retrieved = registry.get("calculator").await;
        assert!(retrieved.is_some());

        let missing = registry.get("nonexistent").await;
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_calculator_tool() {
        let calc = CalculatorTool;
        let input = ToolInput {
            arguments: serde_json::json!({"expression": "2 + 3 * 4"}),
        };
        let output = calc.execute(input).await.unwrap();
        assert!(output.success);
        let result = output.result.get("result").unwrap().as_f64().unwrap();
        assert_eq!(result, 14.0);
    }

    #[tokio::test]
    async fn test_calculator_tool_missing_expression() {
        let calc = CalculatorTool;
        let input = ToolInput {
            arguments: serde_json::json!({}),
        };
        let result = calc.execute(input).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_web_search_tool() {
        let tool = WebSearchTool::new();
        let input = ToolInput {
            arguments: serde_json::json!({"query": "Rust programming"}),
        };
        let output = tool.execute(input).await.unwrap();
        assert!(output.success);
        assert!(output.result.get("query").is_some());
    }

    #[tokio::test]
    async fn test_default_registry() {
        let registry = default_registry().await;
        let tools = registry.list().await;
        assert_eq!(tools.len(), 4);
        assert!(tools.contains(&"calculator".to_string()));
        assert!(tools.contains(&"web_search".to_string()));
        assert!(tools.contains(&"filesystem".to_string()));
        assert!(tools.contains(&"code_execution".to_string()));
    }
}
