//! Observability layer for the Kairo agentic AI orchestrator.
//!
//! This crate provides tracing, metrics, and OpenTelemetry integration used
//! across all workspace members. Call [`init`] once at application startup to
//! configure both tracing and metrics.
//!
//! # Example
//!
//! ```rust,no_run
//! use kairo_telemetry::{init, Meter};
//!
//! let _guard = init().expect("telemetry init failed");
//!
//! let meter = Meter::new();
//! meter.increment_tool_call_count("search");
//! ```

use std::borrow::Cow;
use std::sync::OnceLock;
use std::time::Duration;

pub use metrics;
pub use tracing;

#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    #[error("tracing initialization failed: {0}")]
    TracingInit(String),
    #[error("metrics initialization failed: {0}")]
    MetricsInit(String),
}

pub type Result<T> = std::result::Result<T, TelemetryError>;

#[derive(Debug)]
#[must_use]
pub struct TelemetryGuard;

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        opentelemetry::global::shutdown_tracer_provider();
    }
}

pub fn init() -> Result<TelemetryGuard> {
    init_tracing()?;
    init_metrics()?;
    Ok(TelemetryGuard)
}

pub fn init_tracing() -> Result<()> {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    let result = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .json()
        .with_current_span(true)
        .with_span_list(true)
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .try_init();

    match result {
        Ok(()) => Ok(()),
        Err(e) if e.to_string().contains("already been set") => {
            tracing::debug!("Tracing already initialized, skipping");
            Ok(())
        }
        Err(e) => Err(TelemetryError::TracingInit(e.to_string())),
    }
}

static PROMETHEUS_HANDLE: OnceLock<metrics_exporter_prometheus::PrometheusHandle> = OnceLock::new();

pub fn init_metrics() -> Result<()> {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();

    let handle = recorder.handle();
    let _ = PROMETHEUS_HANDLE.set(handle);

    metrics::set_global_recorder(recorder)
        .map_err(|e| TelemetryError::MetricsInit(e.to_string()))?;

    Ok(())
}

pub fn render_metrics() -> Option<String> {
    PROMETHEUS_HANDLE.get().map(|h| h.render())
}

#[derive(Debug, Clone)]
pub struct Tracer {
    name: &'static str,
}

impl Tracer {
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }

    pub fn start(&self, span_name: &str) -> opentelemetry::global::BoxedSpan {
        use opentelemetry::trace::Tracer;
        opentelemetry::global::tracer(self.name).start(span_name.to_string())
    }
}

pub fn agent_span(agent_id: &str) -> tracing::Span {
    tracing::info_span!("agent_execution", agent.id = %agent_id)
}

pub fn tool_span(tool_name: &str) -> tracing::Span {
    tracing::info_span!("tool_call", tool.name = %tool_name)
}

pub fn provider_span(provider: &str) -> tracing::Span {
    tracing::info_span!("provider_request", provider.name = %provider)
}

pub async fn with_agent_span<F, R>(agent_id: &str, f: F) -> R
where
    F: std::future::Future<Output = R>,
{
    use tracing::Instrument;
    f.instrument(agent_span(agent_id)).await
}

pub async fn with_tool_span<F, R>(tool_name: &str, f: F) -> R
where
    F: std::future::Future<Output = R>,
{
    use tracing::Instrument;
    f.instrument(tool_span(tool_name)).await
}

pub async fn with_provider_span<F, R>(provider: &str, f: F) -> R
where
    F: std::future::Future<Output = R>,
{
    use tracing::Instrument;
    f.instrument(provider_span(provider)).await
}

#[derive(Debug, Clone, Default)]
pub struct Meter;

impl Meter {
    pub fn new() -> Self {
        Self
    }

    pub fn record_agent_execution_duration(&self, agent_id: &str, duration: Duration) {
        metrics::histogram!(
            "agent_execution_duration",
            "agent_id" => Cow::<'static, str>::from(agent_id.to_string())
        )
        .record(duration.as_secs_f64());
    }

    pub fn increment_tool_call_count(&self, tool_name: &str) {
        metrics::counter!(
            "tool_call_count",
            "tool_name" => Cow::<'static, str>::from(tool_name.to_string())
        )
        .increment(1);
    }

    pub fn record_provider_request_latency(&self, provider: &str, latency: Duration) {
        metrics::histogram!(
            "provider_request_latency",
            "provider" => Cow::<'static, str>::from(provider.to_string())
        )
        .record(latency.as_secs_f64());
    }

    pub fn record_token_usage(&self, model: &str, tokens: u64) {
        metrics::counter!(
            "token_usage_total",
            "model" => Cow::<'static, str>::from(model.to_string())
        )
        .increment(tokens);
    }
}
