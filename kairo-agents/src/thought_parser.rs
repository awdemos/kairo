use std::collections::HashMap;

use kairo_core::KairoError;
use serde_json::Value;

use crate::ToolCall;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParsedThought {
    pub thought: String,
    pub action: Option<String>,
    pub observation: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Default)]
pub struct ThoughtParser;

impl ThoughtParser {
    pub fn parse(&self, content: &str) -> Result<ParsedThought, KairoError> {
        let mut thought = String::new();
        let mut action: Option<String> = None;
        let mut observation: Option<String> = None;
        let mut tool_calls = Vec::new();

        let mut current_field: Option<&str> = None;
        let mut buffer = String::new();

        for line in content.lines() {
            if let Some((key, value)) = line.split_once(':') {
                if let Some(field) = current_field {
                    Self::flush(
                        field,
                        &buffer,
                        &mut thought,
                        &mut action,
                        &mut observation,
                        &mut tool_calls,
                    )?;
                }
                current_field = Some(key.trim());
                buffer = value.trim().to_string();
            } else if current_field.is_some() {
                buffer.push('\n');
                buffer.push_str(line);
            }
        }
        if let Some(field) = current_field {
            Self::flush(
                field,
                &buffer,
                &mut thought,
                &mut action,
                &mut observation,
                &mut tool_calls,
            )?;
        }

        Ok(ParsedThought {
            thought,
            action,
            observation,
            tool_calls,
        })
    }

    fn flush(
        field: &str,
        value: &str,
        thought: &mut String,
        action: &mut Option<String>,
        observation: &mut Option<String>,
        tool_calls: &mut Vec<ToolCall>,
    ) -> Result<(), KairoError> {
        match field {
            "Thought" => *thought = value.to_string(),
            "Action" => *action = Some(value.to_string()),
            "Observation" => *observation = Some(value.to_string()),
            "Tool" => {
                let call = Self::parse_tool_call(value)?;
                tool_calls.push(call);
            },
            _ => {},
        }
        Ok(())
    }

    fn parse_tool_call(value: &str) -> Result<ToolCall, KairoError> {
        let (name, args) = value
            .split_once('(')
            .ok_or_else(|| KairoError::Agent("Invalid tool call format: missing '('".into()))?;
        let args = args.strip_suffix(')').ok_or_else(|| {
            KairoError::Agent("Invalid tool call format: missing closing ')'".into())
        })?;

        let mut map = HashMap::new();
        for part in args.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let (k, v) = part
                .split_once('=')
                .ok_or_else(|| KairoError::Agent(format!("Invalid tool argument: {part}")))?;
            map.insert(k.trim().to_string(), Value::String(v.trim().to_string()));
        }

        Ok(ToolCall {
            tool_name: name.trim().to_string(),
            arguments: Value::Object(map.into_iter().collect()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_thought_action_tool_and_observation() {
        let text = "Thought: I need to search\nAction: web_search\nTool: web_search(query=rust)\nObservation: results";
        let thought = ThoughtParser.parse(text).unwrap();
        assert_eq!(thought.thought, "I need to search");
        assert_eq!(thought.action, Some("web_search".to_string()));
        assert_eq!(thought.tool_calls.len(), 1);
        assert_eq!(thought.tool_calls[0].tool_name, "web_search");
    }

    #[test]
    fn finish_action_has_no_tools() {
        let text = "Thought: done\nAction: finish";
        let thought = ThoughtParser.parse(text).unwrap();
        assert_eq!(thought.action, Some("finish".to_string()));
        assert!(thought.tool_calls.is_empty());
    }
}
