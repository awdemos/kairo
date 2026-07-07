use crate::model::ModelId;

impl ModelId {
    pub fn resolve(name: &str) -> Option<Self> {
        match name {
            "gpt-4o" => Some(ModelId::Gpt4o),
            "gpt-4o-mini" => Some(ModelId::Gpt4oMini),
            "gpt-4" => Some(ModelId::Gpt4),
            "gpt-4-turbo" => Some(ModelId::Gpt4Turbo),
            "gpt-3.5-turbo" => Some(ModelId::Gpt3_5Turbo),
            "o1" => Some(ModelId::O1),
            "o1-mini" => Some(ModelId::O1Mini),
            "o3" => Some(ModelId::O3),
            "o3-mini" => Some(ModelId::O3Mini),
            "o4" => Some(ModelId::O4),
            "o4-mini" => Some(ModelId::O4Mini),
            "claude-3-5-sonnet" => Some(ModelId::Claude3_5Sonnet),
            "claude-3-opus" => Some(ModelId::Claude3Opus),
            "claude-3-haiku" => Some(ModelId::Claude3Haiku),
            "claude-3-5-haiku" => Some(ModelId::Claude3_5Haiku),
            "claude-4" => Some(ModelId::Claude4),
            "claude-4-opus" => Some(ModelId::Claude4Opus),
            "gemini-2.0-flash" => Some(ModelId::Gemini2_0Flash),
            "gemini-2.0-pro" => Some(ModelId::Gemini2_0Pro),
            "gemini-2.5-flash" => Some(ModelId::Gemini2_5Flash),
            "gemini-2.5-pro" => Some(ModelId::Gemini2_5Pro),
            "gemini-1.5-pro" => Some(ModelId::Gemini1_5Pro),
            "gemini-1.5-flash" => Some(ModelId::Gemini1_5Flash),
            _ if name.starts_with("gpt-")
                || name.starts_with("o1")
                || name.starts_with("o3")
                || name.starts_with("o4") =>
            {
                Some(ModelId::Custom(name.to_string()))
            }
            _ if name.starts_with("claude-") => Some(ModelId::Custom(name.to_string())),
            _ if name.starts_with("gemini-") => Some(ModelId::Custom(name.to_string())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_known_model_strings() {
        assert_eq!(ModelId::resolve("gpt-4o"), Some(ModelId::Gpt4o));
        assert_eq!(ModelId::resolve("claude-3-5-sonnet"), Some(ModelId::Claude3_5Sonnet));
        assert_eq!(ModelId::resolve("gemini-2.0-flash"), Some(ModelId::Gemini2_0Flash));
    }

    #[test]
    fn custom_models_are_resolved_by_provider_prefix() {
        assert_eq!(ModelId::resolve("gpt-fake"), Some(ModelId::Custom("gpt-fake".into())));
        assert_eq!(ModelId::resolve("claude-fake"), Some(ModelId::Custom("claude-fake".into())));
        assert_eq!(ModelId::resolve("gemini-fake"), Some(ModelId::Custom("gemini-fake".into())));
    }

    #[test]
    fn unknown_prefix_returns_none() {
        assert_eq!(ModelId::resolve("totally-unknown"), None);
    }
}
