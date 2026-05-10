use reqwest::header::{self, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, instrument};

use kairo_core::{
    CompletionOptions, CompletionResponse, KairoError, Message, ModelId, Provider, TokenUsage,
};

/// OpenAI-compatible provider implementation.
pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
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
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|e| KairoError::Provider(format!("Invalid API key: {}", e)))?,
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
            .map_err(|e| KairoError::Provider(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!(status = %status, error = %text, "OpenAI API error");
            return Err(KairoError::Provider(format!(
                "API error {}: {}",
                status, text
            )));
        }

        let openai_response: OpenAiResponse = response
            .json()
            .await
            .map_err(|e| KairoError::Provider(format!("Failed to parse response: {}", e)))?;

        let choice = openai_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| KairoError::Provider("No choices in response".into()))?;

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

/// Anthropic Claude provider implementation.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to build HTTP client"),
            api_key: api_key.into(),
            model: model.into(),
        }
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
            HeaderValue::from_str(&self.api_key)
                .map_err(|e| KairoError::Provider(format!("Invalid API key: {}", e)))?,
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
            .post("https://api.anthropic.com/v1/messages")
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| KairoError::Provider(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!(status = %status, error = %text, "Anthropic API error");
            return Err(KairoError::Provider(format!(
                "API error {}: {}",
                status, text
            )));
        }

        let anthropic_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| KairoError::Provider(format!("Failed to parse response: {}", e)))?;

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
                total_tokens: anthropic_response.usage.input_tokens
                    + anthropic_response.usage.output_tokens,
            },
            model: ModelId::Custom(self.model.clone()),
            finish_reason: anthropic_response.stop_reason.unwrap_or_default(),
        })
        })
    }
}

/// Google Gemini provider implementation.
pub struct GeminiProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl GeminiProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to build HTTP client"),
            api_key: api_key.into(),
            model: model.into(),
        }
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
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .header(header::CONTENT_TYPE, "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| KairoError::Provider(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!(status = %status, error = %text, "Gemini API error");
            return Err(KairoError::Provider(format!(
                "API error {}: {}",
                status, text
            )));
        }

        let gemini_response: GeminiResponse = response
            .json()
            .await
            .map_err(|e| KairoError::Provider(format!("Failed to parse response: {}", e)))?;

        let candidate = gemini_response
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| KairoError::Provider("No candidates in response".into()))?;

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

/// Factory function to create a provider from a ModelId and API key.
pub fn create_provider(model: &ModelId, api_key: &str) -> Result<Arc<dyn Provider>, KairoError> {
    match model {
        ModelId::Gpt4o | ModelId::Gpt4oMini | ModelId::Gpt4 | ModelId::Gpt4Turbo | ModelId::Gpt3_5Turbo | ModelId::O1 | ModelId::O3Mini => {
            Ok(Arc::new(OpenAiProvider::new(api_key, model.to_string())))
        }
        ModelId::Claude3_5Sonnet | ModelId::Claude3Opus | ModelId::Claude3Haiku | ModelId::Claude3_5Haiku => {
            Ok(Arc::new(AnthropicProvider::new(api_key, model.to_string())))
        }
        ModelId::Gemini2_0Flash | ModelId::Gemini1_5Pro | ModelId::Gemini1_5Flash => {
            Ok(Arc::new(GeminiProvider::new(api_key, model.to_string())))
        }
        ModelId::Custom(name) => {
            if name.starts_with("gpt-") || name.starts_with("o1") || name.starts_with("o3") {
                Ok(Arc::new(OpenAiProvider::new(api_key, name.clone())))
            } else if name.starts_with("claude-") {
                Ok(Arc::new(AnthropicProvider::new(api_key, name.clone())))
            } else if name.starts_with("gemini-") {
                Ok(Arc::new(GeminiProvider::new(api_key, name.clone())))
            } else {
                Err(KairoError::Model(format!("Unsupported model: {}", name)))
            }
        }
        _ => Err(KairoError::Model(format!(
            "No provider available for model: {:?}",
            model
        ))),
    }
}


