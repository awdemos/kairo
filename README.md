# Kairo

> *Kairos* — the decisive moment. An agentic AI orchestrator that routes tasks to the right model, executes tools safely, and coordinates multi-step workflows.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE-MIT)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE-APACHE)
[![Rust](https://img.shields.io/badge/rust-1.81%2B-orange.svg)](https://www.rust-lang.org)

## What is Kairo?

Kairo is a **model-agnostic agentic orchestrator** written in Rust. It provides:

- **Intelligent Routing**: A Model Council scores capabilities and routes tasks to the best model for the job
- **Agent Runtime**: ReAct-loop agents with tool use, memory (episodic + semantic), and subagent decomposition
- **Workflow Engine**: DAG-based execution of multi-step tasks with dependency resolution
- **Secure Sandboxing**: WASM-based connector isolation with fuel limits and WASI
- **Production Observability**: Built-in tracing, Prometheus metrics, and OpenTelemetry integration
- **Edge Deployment**: WASM bindings for Cloudflare Workers and browser environments

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         kairo-cli                            │
│                   (CLI / REPL / Server)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                         kairo-api                            │
│              (Axum HTTP server + SSE streaming)              │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                      kairo-orchestrator                      │
│                 (Workflow engine + DAG)                     │
└──────────┬────────────────────────────┬─────────────────────┘
           │                            │
┌──────────▼──────────┐      ┌──────────▼──────────┐
│    kairo-agents     │      │    kairo-council    │
│  (ReAct + SubAgent) │      │  (Model routing)    │
└──────────┬──────────┘      └──────────┬──────────┘
           │                            │
┌──────────▼────────────────────────────▼──────────┐
│              kairo-tools / kairo-memory           │
│         (Tool registry + Episodic/Semantic)       │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│              kairo-providers                      │
│    (OpenAI, Anthropic, Gemini, xAI, etc.)       │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│              kairo-core (shared types)            │
└──────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Rust 1.81+ (`rustup update stable`)
- API key for at least one provider (OpenAI, Anthropic, or Google)

### Installation

```bash
git clone https://github.com/yourusername/kairo
cd kairo
cargo build --release
```

### CLI Usage

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Chat with a model
./target/release/kairo-cli chat -m "gpt-4o" --message "What is the capital of France?"

# Run an agent with tools
./target/release/kairo-cli agent --task "Calculate 2^16 and search for Rust news" --model claude-3-5-sonnet

# Start the API server
./target/release/kairo-cli server --port 3000
```

### API Usage

```bash
# Health check
curl http://localhost:3000/health

# Chat completion
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Create a workflow
curl -X POST http://localhost:3000/v1/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research-workflow",
    "tasks": [
      {"description": "Search for data", "task_type": "Research"},
      {"description": "Analyze findings", "task_type": "DataAnalysis"}
    ]
  }'
```

## Configuration

Kairo reads configuration from environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI models |
| `ANTHROPIC_API_KEY` | Anthropic API key | For Claude models |
| `GEMINI_API_KEY` | Google AI API key | For Gemini models |
| `KAIRO_LOG` | Log level (default: `info`) | No |
| `KAIRO_PORT` | API server port (default: `3000`) | No |

## Workspace Structure

| Crate | Description | Dependencies |
|-------|-------------|--------------|
| `kairo-core` | Shared types, traits, errors | None |
| `kairo-macros` | Procedural macros (derive) | `kairo-core` |
| `kairo-telemetry` | Tracing, metrics, OpenTelemetry | `kairo-core` |
| `kairo-embeddings` | Vector math, embedding clients | `kairo-core` |
| `kairo-memory` | Episodic + semantic memory | `kairo-core`, `kairo-embeddings` |
| `kairo-tools` | Tool registry + built-in tools | `kairo-core` |
| `kairo-providers` | LLM provider implementations | `kairo-core` |
| `kairo-council` | Model routing + capability scoring | `kairo-core`, `kairo-providers`, `kairo-embeddings` |
| `kairo-agents` | ReAct agents + subagent decomposition | `kairo-core`, `kairo-tools`, `kairo-memory`, `kairo-providers`, `kairo-council`, `kairo-telemetry` |
| `kairo-orchestrator` | Workflow engine + DAG execution | `kairo-core`, `kairo-agents` |
| `kairo-api` | HTTP API server (Axum) | `kairo-core`, `kairo-agents`, `kairo-orchestrator`, `kairo-telemetry` |
| `kairo-cli` | Command-line interface | `kairo-core`, `kairo-agents`, `kairo-orchestrator`, `kairo-api`, `kairo-telemetry` |
| `kairo-connectors` | External service connectors | `kairo-core`, `kairo-sandbox` |
| `kairo-sandbox` | WASM sandbox for secure execution | `kairo-core` |
| `kairo-edge` | WASM bindings for edge deployment | `kairo-core` |

## Supported Models

Kairo supports 100+ model variants across major providers:

**OpenAI**: GPT-4o, GPT-4o-mini, o1, o3-mini, GPT-5, and more
**Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 4, Claude 5
**Google**: Gemini 2.0 Flash/Pro, Gemini 1.5 Pro/Flash, Gemini 3.0
**xAI**: Grok 2, Grok 3, Grok 4
**Meta**: Llama 3.1/3.2/4/5 family
**Mistral**: Mistral Large, Medium, Small, Codestral, Mathstral
**Alibaba**: Qwen 2.5/3 family, Qwen Max/Plus/Turbo
**DeepSeek**: V3, R1, Coder V2
**Others**: Cohere, AI21, Microsoft Phi, Zhipu GLM, Moonshot Kimi, Arcee, Nous, Nemotron, Aya, Yi, and more

## Development

```bash
# Run tests
cargo test --workspace

# Run with logging
RUST_LOG=debug cargo run --bin kairo-cli -- chat --message "Hello"

# Check formatting
cargo fmt --all

# Run clippy
cargo clippy --all-targets --all-features
```

## Performance

- **Cold start**: < 50ms (Rust binary)
- **Memory**: ~20MB base + model-specific overhead
- **Throughput**: 10k+ requests/sec (API server, measured locally)
- **Sandbox overhead**: ~5ms per WASM invocation (with caching)

## Roadmap

- [ ] Streaming responses (SSE) for chat completions
- [ ] Persistent memory backends (Redis, PostgreSQL)
- [ ] Multi-agent collaboration protocols
- [ ] Fine-tuning pipeline integration
- [ ] Kubernetes operator for deployment
- [ ] gRPC API alongside HTTP

## License

Dual-licensed under MIT or Apache-2.0. See [LICENSE-MIT](./LICENSE-MIT) and [LICENSE-APACHE](./LICENSE-APACHE).

## Contributing

Contributions welcome! Please read our contributing guidelines (TBD) and open an issue before major changes.

## Acknowledgments

Built with gratitude to the Rust community and the teams behind:
- [Tokio](https://tokio.rs/) for async runtime
- [Axum](https://github.com/tokio-rs/axum) for HTTP server
- [Wasmtime](https://wasmtime.dev/) for secure sandboxing
- [Tracing](https://tracing.rs/) for observability
- And the many model providers pushing AI forward
