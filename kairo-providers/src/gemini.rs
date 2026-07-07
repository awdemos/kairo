use reqwest::header::{self};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, instrument};

use kairo_core::{
    CompletionOptions, CompletionResponse, KairoError, Message, ModelId, Provider, ProviderError,
    TokenUsage,
};

/// Google Gemini provider implementation.
pub struct GeminiProvider {
    pub(crate) client: reqwest::Client,
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) model: String,
}

impl GeminiProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to build HTTP client"),
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            model: model.into(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    usage_metadata: Option<GeminiUsage>,
    model_version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContentResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GeminiContentResponse {
    parts: Vec<GeminiPartResponse>,
    role: String,
}

#[derive(Debug, Deserialize)]
struct GeminiPartResponse {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiUsage {
    prompt_token_count: u64,
    candidates_token_count: u64,
    total_token_count: u64,
}

impl Provider for GeminiProvider {
    #[instrument(skip(self, messages, options))]
    fn complete<'a>(
        &'a self,
        messages: Vec<Message>,
        options: CompletionOptions,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<CompletionResponse, KairoError>> + Send + 'a>> {
        Box::pin(async move {
            let contents: Vec<GeminiContent> = messages
                .into_iter()
                .map(|m| GeminiContent {
                    role: match m.role {
                        kairo_core::Role::System => "user".to_string(),
                        kairo_core::Role::User => "user".to_string(),
                        kairo_core::Role::Assistant => "model".to_string(),
                        kairo_core::Role::Tool => "user".to_string(),
                    },
                    parts: vec![GeminiPart { text: m.content }],
                })
                .collect();

            let request = GeminiRequest {
                contents,
                generation_config: Some(GeminiGenerationConfig {
                    temperature: options.temperature,
                    max_output_tokens: options.max_tokens,
                    top_p: options.top_p,
                }),
            };

            debug!(model = %self.model, "sending Gemini completion request");

            let url = format!(
                "{}/models/{}:generateContent?key={}",
                self.base_url, self.model, self.api_key
            );

            let response = self
                .client
                .post(&url)
                .header(header::CONTENT_TYPE, "application/json")
                .json(&request)
                .send()
                .await
                .map_err(|e| {
                    KairoError::Provider(
                        ProviderError::new("gemini", &self.model).with_message(format!("Request failed: {}", e)),
                    )
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                error!(status = %status, error = %text, "Gemini API error");
                return Err(KairoError::Provider(
                    ProviderError::new("gemini", &self.model)
                        .with_status(status.as_u16())
                        .with_retryable(crate::is_retryable(status.as_u16()))
                        .with_message(text),
                ));
            }

            let gemini_response: GeminiResponse = response.json().await.map_err(|e| {
                KairoError::Provider(
                    ProviderError::new("gemini", &self.model).with_message(format!("Failed to parse response: {}", e)),
                )
            })?;

            let candidate = gemini_response
                .candidates
                .into_iter()
                .next()
                .ok_or_else(|| {
                    KairoError::Provider(
                        ProviderError::new("gemini", &self.model).with_message("No candidates in response"),
                    )
                })?;

            let content_text = candidate
                .content
                .parts
                .into_iter()
                .map(|p| p.text)
                .collect::<Vec<_>>()
                .join("");

            let usage = gemini_response.usage_metadata.unwrap_or(GeminiUsage {
                prompt_token_count: 0,
                candidates_token_count: 0,
                total_token_count: 0,
            });

            Ok(CompletionResponse {
                content: content_text,
                usage: TokenUsage {
                    prompt_tokens: usage.prompt_token_count,
                    completion_tokens: usage.candidates_token_count,
                    total_tokens: usage.total_token_count,
                },
                model: ModelId::Custom(self.model.clone()),
                finish_reason: candidate.finish_reason.unwrap_or_default(),
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
    async fn test_gemini_provider_parses_successful_response() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/gemini-1\.5-flash:generateContent(\?.*)?$".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"candidates":[{"content":{"parts":[{"text":"hello"}],"role":"model"},"finish_reason":"STOP"}],"usage_metadata":{"prompt_token_count":5,"candidates_token_count":2,"total_token_count":7},"model_version":"gemini-1.5-flash"}"#)
            .create_async()
            .await;

        let provider = GeminiProvider::new("test-key", "gemini-1.5-flash").with_base_url(format!("{}/v1beta", server.url()));
        let response = provider
            .complete(vec![Message { role: Role::User, content: "hi".into(), name: None, timestamp: Utc::now() }], CompletionOptions::default())
            .await
            .unwrap();

        assert_eq!(response.content, "hello");
        assert_eq!(response.usage.total_tokens, 7);
    }
}
