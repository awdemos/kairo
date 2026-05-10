use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KairoConfig {
    pub providers: ProviderConfig,
    pub api: ApiConfig,
    pub telemetry: TelemetryConfig,
    pub tools: ToolConfig,
}

impl Default for KairoConfig {
    fn default() -> Self {
        Self {
            providers: ProviderConfig::default(),
            api: ApiConfig::default(),
            telemetry: TelemetryConfig::default(),
            tools: ToolConfig::default(),
        }
    }
}

impl KairoConfig {
    pub fn from_env() -> Self {
        Self {
            providers: ProviderConfig::from_env(),
            api: ApiConfig::from_env(),
            telemetry: TelemetryConfig::from_env(),
            tools: ToolConfig::from_env(),
        }
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, crate::KairoError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| crate::KairoError::Internal(format!("Failed to read config file: {}", e)))?;
        toml::from_str(&contents)
            .map_err(|e| crate::KairoError::Internal(format!("Failed to parse config file: {}", e)))
    }

    pub fn load() -> Self {
        if let Ok(config) = Self::from_file("kairo.toml") {
            return config;
        }
        if let Some(home) = dirs::home_dir() {
            let path = home.join(".config").join("kairo").join("config.toml");
            if let Ok(config) = Self::from_file(&path) {
                return config;
            }
        }
        Self::from_env()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub xai_api_key: Option<String>,
    pub default_model: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            openai_api_key: None,
            anthropic_api_key: None,
            gemini_api_key: None,
            xai_api_key: None,
            default_model: "gpt-4o".into(),
            timeout_seconds: 120,
            max_retries: 3,
        }
    }
}

impl ProviderConfig {
    pub fn from_env() -> Self {
        Self {
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            anthropic_api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            gemini_api_key: std::env::var("GEMINI_API_KEY").ok(),
            xai_api_key: std::env::var("XAI_API_KEY").ok(),
            default_model: std::env::var("KAIRO_DEFAULT_MODEL").unwrap_or_else(|_| "gpt-4o".into()),
            timeout_seconds: std::env::var("KAIRO_TIMEOUT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(120),
            max_retries: std::env::var("KAIRO_MAX_RETRIES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
    pub request_timeout: u64,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 3000,
            cors_origins: vec!["*".into()],
            request_timeout: 300,
        }
    }
}

impl ApiConfig {
    pub fn from_env() -> Self {
        Self {
            host: std::env::var("KAIRO_API_HOST").unwrap_or_else(|_| "127.0.0.1".into()),
            port: std::env::var("KAIRO_API_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3000),
            cors_origins: std::env::var("KAIRO_CORS_ORIGINS")
                .ok()
                .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_else(|| vec!["*".into()]),
            request_timeout: std::env::var("KAIRO_REQUEST_TIMEOUT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(300),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    pub log_level: String,
    pub metrics_enabled: bool,
    pub metrics_port: u16,
    pub otlp_endpoint: Option<String>,
    pub otlp_enabled: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            log_level: "info".into(),
            metrics_enabled: true,
            metrics_port: 9090,
            otlp_endpoint: None,
            otlp_enabled: false,
        }
    }
}

impl TelemetryConfig {
    pub fn from_env() -> Self {
        Self {
            log_level: std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
            metrics_enabled: std::env::var("KAIRO_METRICS_ENABLED")
                .ok()
                .map(|s| s == "true" || s == "1")
                .unwrap_or(true),
            metrics_port: std::env::var("KAIRO_METRICS_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9090),
            otlp_endpoint: std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok(),
            otlp_enabled: std::env::var("KAIRO_OTLP_ENABLED")
                .ok()
                .map(|s| s == "true" || s == "1")
                .unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    pub serpapi_key: Option<String>,
    pub brave_api_key: Option<String>,
    pub sandbox_enabled: bool,
    pub max_sandbox_memory_mb: usize,
    pub max_sandbox_fuel: u64,
    pub allowed_tools: Vec<String>,
    pub tool_timeouts: HashMap<String, u64>,
}

impl Default for ToolConfig {
    fn default() -> Self {
        let mut tool_timeouts = HashMap::new();
        tool_timeouts.insert("web_search".into(), 30);
        tool_timeouts.insert("code_execution".into(), 60);
        tool_timeouts.insert("file_system".into(), 10);

        Self {
            serpapi_key: None,
            brave_api_key: None,
            sandbox_enabled: false,
            max_sandbox_memory_mb: 64,
            max_sandbox_fuel: 10_000_000,
            allowed_tools: vec!["*".into()],
            tool_timeouts,
        }
    }
}

impl ToolConfig {
    pub fn from_env() -> Self {
        let mut defaults = Self::default();
        defaults.serpapi_key = std::env::var("SERPAPI_KEY").ok();
        defaults.brave_api_key = std::env::var("BRAVE_API_KEY").ok();
        defaults.sandbox_enabled = std::env::var("KAIRO_SANDBOX_ENABLED")
            .ok()
            .map(|s| s == "true" || s == "1")
            .unwrap_or(false);
        defaults
    }
}
