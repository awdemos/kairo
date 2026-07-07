use reqwest::header::{self, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, instrument};

use kairo_core::{
    CompletionOptions, CompletionResponse, KairoError, Message, ModelId, Provider, ProviderError,
    TokenUsage,
};

/// Anthropic Claude provider implementation.
pub struct AnthropicProvider {
    pub(crate) client: reqwest::Client,
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) model: String,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to build HTTP client"),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            model: model.into(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    usage: AnthropicUsage,
    stop_reason: Option<String>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
}

impl Provider for AnthropicProvider {
    #[instrument(skip(self, messages, options))]
    fn complete<'a>(
        &'a self,
        messages: Vec<Message>,
        options: CompletionOptions,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<CompletionResponse, KairoError>> + Send + 'a>> {
        Box::pin(async move {
            let anthropic_messages: Vec<AnthropicMessage> = messages
                .iter()
                .filter(|m| m.role != kairo_core::Role::System)
                .map(|m| AnthropicMessage {
                    role: match m.role {
                        kairo_core::Role::System => "system".to_string(),
                        kairo_core::Role::User => "user".to_string(),
                        kairo_core::Role::Assistant => "assistant".to_string(),
                        kairo_core::Role::Tool => "user".to_string(),
                    },
                    content: m.content.clone(),
                })
                .collect();

            let system_prompt = messages
                .iter()
                .find(|m| m.role == kairo_core::Role::System)
                .map(|m| m.content.clone());

            let request = AnthropicRequest {
                model: self.model.clone(),
                messages: anthropic_messages,
                max_tokens: options.max_tokens,
                temperature: options.temperature,
                top_p: options.top_p,
                system: system_prompt,
            };

            let mut headers = HeaderMap::new();
            headers.insert(
                "x-api-key",
                HeaderValue::from_str(&self.api_key).map_err(|e| {
                    KairoError::Provider(
                        ProviderError::new("anthropic", &self.model).with_message(format!("Invalid API key: {}", e)),
                    )
                })?,
            );
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            headers.insert(
                "anthropic-version",
                HeaderValue::from_static("2023-06-01"),
            );

            debug!(model = %self.model, "sending Anthropic completion request");

            let response = self
                .client
                .post(format!("{}/messages", self.base_url))
                .headers(headers)
                .json(&request)
                .send()
                .await
                .map_err(|e| {
                    KairoError::Provider(
                        ProviderError::new("anthropic", &self.model).with_message(format!("Request failed: {}", e)),
                    )
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                error!(status = %status, error = %text, "Anthropic API error");
                return Err(KairoError::Provider(
                    ProviderError::new("anthropic", &self.model)
                        .with_status(status.as_u16())
                        .with_retryable(crate::is_retryable(status.as_u16()))
                        .with_message(text),
                ));
            }

            let anthropic_response: AnthropicResponse = response.json().await.map_err(|e| {
                KairoError::Provider(
                    ProviderError::new("anthropic", &self.model).with_message(format!("Failed to parse response: {}", e)),
                )
            })?;

            let content_text = anthropic_response
                .content
                .into_iter()
                .filter(|c| c.content_type == "text")
                .map(|c| c.text)
                .collect::<Vec<_>>()
                .join("");

            Ok(CompletionResponse {
                content: content_text,
                usage: TokenUsage {
                    prompt_tokens: anthropic_response.usage.input_tokens,
                    completion_tokens: anthropic_response.usage.output_tokens,
                    total_tokens: anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
                },
                model: ModelId::Custom(self.model.clone()),
                finish_reason: anthropic_response.stop_reason.unwrap_or_default(),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use kairo_core::{CompletionOptions, Message, Role};

    #[tokio::test]
    async fn test_anthropic_provider_parses_successful_response() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/v1/messages")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"content":[{"type":"text","text":"hello"}],"usage":{"input_tokens":5,"output_tokens":2},"stop_reason":"end_turn","model":"claude-3-5-sonnet-20240620"}"#)
            .create_async()
            .await;

        let provider = AnthropicProvider::new("test-key", "claude-3-5-sonnet-20240620").with_base_url(format!("{}/v1", server.url()));
        let response = provider
            .complete(vec![Message { role: Role::User, content: "hi".into(), name: None, timestamp: Utc::now() }], CompletionOptions::default())
            .await
            .unwrap();

        assert_eq!(response.content, "hello");
        assert_eq!(response.usage.total_tokens, 7);
    }
}
