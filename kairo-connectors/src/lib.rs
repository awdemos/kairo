use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::future::Future;
use tracing::{debug, instrument};

use kairo_core::{Action, Connector, Data, KairoError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConnectorConfig {
    pub base_url: String,
    pub headers: std::collections::HashMap<String, String>,
    pub timeout_secs: u64,
}

pub struct HttpConnector {
    client: reqwest::Client,
    config: HttpConnectorConfig,
}

impl HttpConnector {
    pub fn new(config: HttpConnectorConfig) -> Result<Self, KairoError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| KairoError::Connector(format!("Failed to build HTTP client: {}", e)))?;

        Ok(Self { client, config })
    }
}

impl Connector for HttpConnector {
    fn invoke<'a>(&'a self, action: Action) -> Pin<Box<dyn Future<Output = Result<Data, KairoError>> + Send + 'a>> {
        Box::pin(async move {
            let url = format!("{}{}", self.config.base_url, action.operation);

            let mut request = self.client.request(
                reqwest::Method::POST,
                &url,
            );

            for (key, value) in &self.config.headers {
                request = request.header(key, value);
            }

            request = request.json(&action.parameters);

            debug!(url = %url, "sending HTTP connector request");

            let response = request.send().await
                .map_err(|e| KairoError::Connector(format!("HTTP request failed: {}", e)))?;

            if !response.status().is_success() {
                return Err(KairoError::Connector(format!(
                    "HTTP error: {}",
                    response.status()
                )));
            }

            let content = response.json().await
                .map_err(|e| KairoError::Connector(format!("Failed to parse response: {}", e)))?;

            Ok(Data { content })
        })
    }
}

pub struct DatabaseConnector {
    connection_string: String,
}

impl DatabaseConnector {
    pub fn new(connection_string: impl Into<String>) -> Self {
        Self {
            connection_string: connection_string.into(),
        }
    }
}

impl Connector for DatabaseConnector {
    fn invoke<'a>(&'a self, action: Action) -> Pin<Box<dyn Future<Output = Result<Data, KairoError>> + Send + 'a>> {
        Box::pin(async move {
            debug!(operation = %action.operation, "database connector placeholder");
            Ok(Data {
                content: serde_json::json!({
                    "status": "placeholder",
                    "operation": action.operation,
                    "note": "Implement with actual database driver (sqlx, tokio-postgres, etc.)"
                }),
            })
        })
    }
}

pub struct FileConnector;

impl Connector for FileConnector {
    fn invoke<'a>(&'a self, action: Action) -> Pin<Box<dyn Future<Output = Result<Data, KairoError>> + Send + 'a>> {
        Box::pin(async move {
            let operation = action.parameters.get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("read");

            let path = action.parameters.get("path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");

            debug!(operation = %operation, path = %path, "file connector");

            match operation {
                "read" => {
                    let content = tokio::fs::read_to_string(path).await
                        .map_err(|e| KairoError::Connector(format!("File read failed: {}", e)))?;
                    Ok(Data {
                        content: serde_json::json!({ "content": content }),
                    })
                }
                "write" => {
                    let content = action.parameters.get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    tokio::fs::write(path, content).await
                        .map_err(|e| KairoError::Connector(format!("File write failed: {}", e)))?;
                    Ok(Data {
                        content: serde_json::json!({ "written": true }),
                    })
                }
                _ => Err(KairoError::Connector(format!("Unknown file operation: {}", operation))),
            }
        })
    }
}

pub fn create_http_connector(base_url: impl Into<String>) -> Result<HttpConnector, KairoError> {
    HttpConnector::new(HttpConnectorConfig {
        base_url: base_url.into(),
        headers: std::collections::HashMap::new(),
        timeout_secs: 30,
    })
}
