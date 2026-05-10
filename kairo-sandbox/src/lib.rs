//! kairo-sandbox — secure WASM execution environment for Kairo
//!
//! Provides resource-limited, isolated execution of untrusted code via
//! wasmtime and the WASI Preview 2 component model.

use std::{
    collections::HashMap,
    sync::Arc,
    time::Duration,
};

use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, instrument, trace};
use wasmtime::{
    component::{Component, Linker, Val},
    Config, Engine, Store, StoreLimits, StoreLimitsBuilder,
};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView, ResourceTable, DirPerms, FilePerms};

pub use kairo_core::{Action, Connector, Data, KairoError};

/// Configuration for sandbox resource limits and WASI capabilities.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Maximum linear memory per instance in bytes (default: 64 MiB).
    pub memory_max: usize,
    /// Maximum fuel (Wasm operations) per execution (default: 10_000_000).
    pub fuel_limit: u64,
    /// Maximum wall-clock execution time (default: 30s).
    pub execution_timeout: Duration,
    /// Maximum number of concurrent executions (default: 10).
    pub max_concurrent: usize,
    /// Allow WASI stdin inheritance.
    pub wasi_stdin: bool,
    /// Allow WASI stdout inheritance.
    pub wasi_stdout: bool,
    /// Allow WASI stderr inheritance.
    pub wasi_stderr: bool,
    /// Allow WASI clock access.
    pub wasi_clocks: bool,
    /// Allow WASI random access.
    pub wasi_random: bool,
    /// Allow TCP socket usage (default: false — blanket deny).
    pub wasi_tcp: bool,
    /// Allow UDP socket usage (default: false — blanket deny).
    pub wasi_udp: bool,
    /// Allow IP name lookup (default: false).
    pub wasi_ip_name_lookup: bool,
    /// Allowed filesystem preopens as `(host_path, guest_path)` tuples.
    pub wasi_preopens: Vec<(String, String)>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            memory_max: 64 * 1024 * 1024,
            fuel_limit: 10_000_000,
            execution_timeout: Duration::from_secs(30),
            max_concurrent: 10,
            wasi_stdin: false,
            wasi_stdout: true,
            wasi_stderr: true,
            wasi_clocks: true,
            wasi_random: false,
            wasi_tcp: false,
            wasi_udp: false,
            wasi_ip_name_lookup: false,
            wasi_preopens: Vec::new(),
        }
    }
}

/// Per-store state that holds the WASI context and resource limits.
pub struct SandboxState {
    wasi_ctx: WasiCtx,
    resource_table: ResourceTable,
    limits: StoreLimits,
}

impl WasiView for SandboxState {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }

    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi_ctx
    }
}

impl SandboxState {
    fn new(config: &SandboxConfig) -> Result<Self, KairoError> {
        let mut builder = WasiCtxBuilder::new();

        if config.wasi_stdin {
            builder.inherit_stdin();
        }
        if config.wasi_stdout {
            builder.inherit_stdout();
        }
        if config.wasi_stderr {
            builder.inherit_stderr();
        }
        builder.allow_tcp(config.wasi_tcp);
        builder.allow_udp(config.wasi_udp);
        builder.allow_ip_name_lookup(config.wasi_ip_name_lookup);

        for (host_path, guest_path) in &config.wasi_preopens {
            builder
                .preopened_dir(
                    host_path,
                    guest_path,
                    DirPerms::READ,
                    FilePerms::READ,
                )
                .map_err(|e| {
                    KairoError::Sandbox(format!(
                        "preopen failed for {}: {}",
                        host_path, e
                    ))
                })?;
        }

        let wasi_ctx = builder.build();

        let limits = StoreLimitsBuilder::new()
            .memory_size(config.memory_max)
            .instances(1)
            .tables(1)
            .memories(1)
            .build();

        Ok(Self {
            wasi_ctx,
            resource_table: ResourceTable::new(),
            limits,
        })
    }
}

/// Secure WASM sandbox for executing untrusted WebAssembly components.
///
/// WASM binaries are compiled once and cached by content hash.  Execution is
/// governed by fuel-based operation limiting, linear-memory bounds, and a
/// tokio-backed concurrency semaphore.
pub struct Sandbox {
    engine: Engine,
    linker: Linker<SandboxState>,
    module_cache: Arc<RwLock<HashMap<[u8; 32], Arc<Component>>>>,
    semaphore: Arc<Semaphore>,
    config: SandboxConfig,
}

impl Sandbox {
    /// Create a new sandbox with the provided configuration.
    pub fn new(config: SandboxConfig) -> Result<Self, KairoError> {
        let mut wasm_config = Config::new();
        wasm_config.cranelift_opt_level(wasmtime::OptLevel::Speed);
        wasm_config.consume_fuel(true);
        wasm_config.memory_init_cow(true);
        wasm_config.wasm_component_model(true);

        let engine = Engine::new(&wasm_config)
            .map_err(|e| KairoError::Sandbox(format!("engine creation failed: {}", e)))?;

        let mut linker = Linker::<SandboxState>::new(&engine);
        wasmtime_wasi::add_to_linker_sync(&mut linker)
            .map_err(|e| KairoError::Sandbox(format!("WASI linker setup failed: {}", e)))?;

        Ok(Self {
            engine,
            linker,
            module_cache: Arc::new(RwLock::new(HashMap::new())),
            semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
            config,
        })
    }

    /// Execute a WASM component function with resource limits.
    ///
    /// The `wasm_bytes` are compiled once and cached by BLAKE3 content hash.
    /// The guest is expected to expose a component function named `function`
    /// that takes a single `list<u8>` parameter and returns `list<u8>`.
    #[instrument(skip(self, wasm_bytes, input), fields(function = %function))]
    pub async fn execute(
        &self,
        wasm_bytes: Vec<u8>,
        function: &str,
        input: Vec<u8>,
    ) -> Result<Vec<u8>, KairoError> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| KairoError::Sandbox(format!("concurrency limit error: {}", e)))?;

        let hash = *blake3::hash(&wasm_bytes).as_bytes();

        let component = {
            let cache = self.module_cache.read().await;
            if let Some(cached) = cache.get(&hash) {
                trace!(hash = %hex_encode(hash), "using cached component");
                cached.clone()
            } else {
                drop(cache);
                debug!(hash = %hex_encode(hash), bytes = wasm_bytes.len(), "compiling component");
                let component = Component::new(&self.engine, &wasm_bytes)
                    .map_err(|e| KairoError::Sandbox(format!("WASM compilation failed: {}", e)))?;
                let component = Arc::new(component);
                let mut cache = self.module_cache.write().await;
                cache.insert(hash, component.clone());
                component
            }
        };

        let engine = self.engine.clone();
        let linker = self.linker.clone();
        let config = self.config.clone();
        let function = function.to_string();

        let result = tokio::time::timeout(
            config.execution_timeout,
            tokio::task::spawn_blocking(move || {
                let state = SandboxState::new(&config)?;
                let mut store = Store::new(&engine, state);
                store
                    .set_fuel(config.fuel_limit)
                    .map_err(|e| KairoError::Sandbox(format!("fuel setup failed: {}", e)))?;
                store.limiter(|state| &mut state.limits);

                let instance = linker
                    .instantiate(&mut store, &component)
                    .map_err(|e| KairoError::Sandbox(format!("instantiation failed: {}", e)))?;

                let func = instance
                    .get_func(&mut store, &function)
                    .ok_or_else(|| {
                        KairoError::Sandbox(format!("function '{}' not found", function))
                    })?;

                let params = vec![Val::List(
                    input.into_iter().map(Val::U8).collect(),
                )];
                let mut results = vec![Val::Bool(false)];

                func.call(&mut store, &params, &mut results)
                    .map_err(|e| KairoError::Sandbox(format!("execution failed: {}", e)))?;

                func.post_return(&mut store)
                    .map_err(|e| KairoError::Sandbox(format!("post-return failed: {}", e)))?;

                match &results[0] {
                    Val::List(list) => {
                        let bytes: Vec<u8> = list
                            .iter()
                            .map(|v| match v {
                                Val::U8(b) => *b,
                                _ => 0,
                            })
                            .collect();
                        Ok(bytes)
                    }
                    _ => Err(KairoError::Sandbox(
                        "unexpected return type from sandbox".into(),
                    )),
                }
            }),
        )
        .await;

        match result {
            Ok(Ok(Ok(bytes))) => Ok(bytes),
            Ok(Ok(Err(e))) => Err(e),
            Ok(Err(join_err)) => Err(KairoError::Sandbox(format!(
                "execution task panicked: {}",
                join_err
            ))),
            Err(_) => Err(KairoError::Sandbox("execution timed out".into())),
        }
    }
}

/// A [`Connector`] implementation that runs connector logic inside the sandbox.
pub struct SandboxedConnector {
    sandbox: Arc<Sandbox>,
    wasm_bytes: Vec<u8>,
    function: String,
}

impl SandboxedConnector {
    /// Create a new sandboxed connector.
    pub fn new(
        sandbox: Arc<Sandbox>,
        wasm_bytes: Vec<u8>,
        function: impl Into<String>,
    ) -> Self {
        Self {
            sandbox,
            wasm_bytes,
            function: function.into(),
        }
    }
}

impl Connector for SandboxedConnector {
    async fn invoke(&self, action: Action) -> Result<Data, KairoError> {
        let input = serde_json::to_vec(&action)
            .map_err(|e| KairoError::Sandbox(format!("action serialization failed: {}", e)))?;

        let output = self
            .sandbox
            .execute(self.wasm_bytes.clone(), &self.function, input)
            .await?;

        let data = serde_json::from_slice(&output)
            .map_err(|e| KairoError::Sandbox(format!("data deserialization failed: {}", e)))?;

        Ok(data)
    }
}

fn hex_encode(bytes: [u8; 32]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(64);
    for b in bytes {
        write!(&mut s, "{:02x}", b).unwrap();
    }
    s
}
