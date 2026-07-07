use reqwest::header::{self, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, instrument};

use kairo_core::{
    CompletionOptions, CompletionResponse, KairoError, Message, ModelId, Provider, ProviderError,
    TokenUsage,
};

/// OpenAI-compatible provider implementation.
pub struct OpenAiProvider {
    pub(crate) client: reqwest::Client,
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) model: String,
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to build HTTP client"),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: model.into(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAiMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: OpenAiUsage,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

impl Provider for OpenAiProvider {
    #[instrument(skip(self, messages, options))]
    fn complete<'a>(
        &'a self,
        messages: Vec<Message>,
        options: CompletionOptions,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<CompletionResponse, KairoError>> + Send + 'a>> {
        Box::pin(async move {
            let openai_messages: Vec<OpenAiMessage> = messages
                .into_iter()
                .map(|m| OpenAiMessage {
                    role: match m.role {
                        kairo_core::Role::System => "system".to_string(),
                        kairo_core::Role::User => "user".to_string(),
                        kairo_core::Role::Assistant => "assistant".to_string(),
                        kairo_core::Role::Tool => "tool".to_string(),
                    },
                    content: m.content,
                    name: m.name,
                })
                .collect();

            let request = OpenAiRequest {
                model: self.model.clone(),
                messages: openai_messages,
                temperature: options.temperature,
                max_tokens: options.max_tokens,
                top_p: options.top_p,
                presence_penalty: options.presence_penalty,
                frequency_penalty: options.frequency_penalty,
                stop: options.stop_sequences,
            };

            let mut headers = HeaderMap::new();
            headers.insert(
                header::AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", self.api_key)).map_err(|e| {
                    KairoError::Provider(
                        ProviderError::new("openai", &self.model)
                            .with_message(format!("Invalid API key: {}", e)),
                    )
                })?,
            );
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );

            debug!(model = %self.model, "sending OpenAI completion request");

            let response = self
                .client
                .post(format!("{}/chat/completions", self.base_url))
                .headers(headers)
                .json(&request)
                .send()
                .await
                .map_err(|e| {
                    KairoError::Provider(
                        ProviderError::new("openai", &self.model)
                            .with_message(format!("Request failed: {}", e)),
                    )
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                error!(status = %status, error = %text, "OpenAI API error");
                return Err(KairoError::Provider(
                    ProviderError::new("openai", &self.model)
                        .with_status(status.as_u16())
                        .with_retryable(crate::is_retryable(status.as_u16()))
                        .with_message(text),
                ));
            }

            let openai_response: OpenAiResponse = response.json().await.map_err(|e| {
                KairoError::Provider(
                    ProviderError::new("openai", &self.model)
                        .with_message(format!("Failed to parse response: {}", e)),
                )
            })?;

            let choice = openai_response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| {
                    KairoError::Provider(
                        ProviderError::new("openai", &self.model).with_message("No choices in response"),
                    )
                })?;

            Ok(CompletionResponse {
                content: choice.message.content,
                usage: TokenUsage {
                    prompt_tokens: openai_response.usage.prompt_tokens,
                    completion_tokens: openai_response.usage.completion_tokens,
                    total_tokens: openai_response.usage.total_tokens,
                },
                model: ModelId::Custom(self.model.clone()),
                finish_reason: choice.finish_reason.unwrap_or_default(),
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
    async fn test_openai_provider_parses_successful_response() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"choices":[{"message":{"role":"assistant","content":"hello"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7},"model":"gpt-4o"}"#)
            .create_async()
            .await;

        let provider = OpenAiProvider::new("test-key", "gpt-4o").with_base_url(format!("{}/v1", server.url()));
        let response = provider
            .complete(vec![Message { role: Role::User, content: "hi".into(), name: None, timestamp: Utc::now() }], CompletionOptions::default())
            .await
            .unwrap();

        assert_eq!(response.content, "hello");
        assert_eq!(response.usage.total_tokens, 7);
    }
}
