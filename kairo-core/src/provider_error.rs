#[derive(Debug, Clone, PartialEq)]
pub struct ProviderIdentity {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderError {
    pub identity: ProviderIdentity,
    pub status: Option<u16>,
    pub retryable: bool,
    pub message: String,
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}) [{}]: {}",
            self.identity.provider,
            self.identity.model,
            self.status
                .map_or_else(|| "no status".to_string(), |s| s.to_string()),
            self.message,
        )
    }
}

impl ProviderError {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            identity: ProviderIdentity {
                provider: provider.into(),
                model: model.into(),
            },
            status: None,
            retryable: false,
            message: String::new(),
        }
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_records_identity_and_retryability() {
        let err = ProviderError::new("openai", "gpt-4o")
            .with_status(429)
            .with_retryable(true)
            .with_message("rate limited");
        assert_eq!(err.identity.provider, "openai");
        assert_eq!(err.identity.model, "gpt-4o");
        assert_eq!(err.status, Some(429));
        assert!(err.retryable);
        assert_eq!(err.message, "rate limited");
    }
}
