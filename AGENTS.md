# Agent Guidelines for Kairo

Kairo is a model-agnostic agentic AI orchestrator written in Rust. It provides ReAct-loop agents, a model-routing council, DAG workflows, memory, and WASM-sandboxed connectors.

## Project Layout

```
Cargo.toml                  # Workspace definition
kairo-core/                 # Shared types and utilities
kairo-providers/            # LLM provider integrations
kairo-tools/                # Tool registry
kairo-memory/               # Episodic and semantic memory
kairo-agents/               # ReAct / subagent runtime
kairo-council/              # Model routing
kairo-orchestrator/         # DAG workflow engine
kairo-api/                  # Axum HTTP server
kairo-cli/                  # CLI / REPL
kairo-edge/                 # WASM bindings
kairo-tests/                # Integration tests
xtask/                      # Workspace automation helpers
```

## Critical Rules

1. **Rust 1.81+ is required.** The workspace `rust-version` is the floor; do not use newer syntax that would break MSRV without updating it.
2. **Keep clippy clean.** `clippy.toml` enables `clippy::all`, `pedantic`, and `cargo` by default and warns on `unwrap_used`, `expect_used`, `panic`, `todo`, and `dbg_macro`.
3. **Errors must be typed.** Prefer `thiserror` enums over `anyhow` in library crates; `anyhow` is acceptable in CLI/edge entry points.
4. **WASI/WASM isolation is a security boundary.** Connector crates must not break sandboxing; fuel limits and capability gating must remain.
5. **Provider keys are never committed.** Use environment variables and the config loader; verify no secrets are added to tests or fixtures.

## Build Commands

Prerequisites: Rust 1.81+, optionally `wasm32-wasi` target for edge builds.

```bash
# Build the whole workspace
cargo build --release

# Build CLI only
cargo build -p kairo-cli --release

# Build API server
cargo build -p kairo-api --release

# WASM edge build
cargo build -p kairo-edge --target wasm32-wasi --release
```

## Test Commands

```bash
# Run workspace tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p kairo-agents

# Run with tracing output for debugging
RUST_LOG=debug cargo test -p kairo-orchestrator -- --nocapture
```

## Lint / Format

```bash
# Format
cargo fmt --all

# Clippy (strict — matching CI)
cargo clippy --all-targets --all-features --workspace -- -D warnings

# Check only
cargo check --workspace
```

## Common Operations

```bash
# CLI chat example
export OPENAI_API_KEY=sk-...
./target/release/kairo-cli chat -m gpt-4o --message "Hello"

# API server
./target/release/kairo-api
```

## Gotchas

- The workspace uses the 2021 edition and resolver = "2".
- `kairo-tests` holds integration tests; do not put slow/network tests in unit-test crates.
- `xtask` helpers may assume `cargo` and `just` conventions; read `xtask/src/main.rs` before adding tasks.
