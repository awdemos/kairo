#![allow(async_fn_in_trait)]

use std::future::Future;
use std::pin::Pin;

use crate::context::Context;
use crate::error::KairoError;
use crate::types::{
    Action, CompletionOptions, CompletionResponse, CostEstimate, Data, Message, Output,
    TaskVector, ToolInput, ToolOutput,
};

pub trait Routable: Send + Sync {
    fn task_vector(&self) -> TaskVector;
    fn estimated_cost(&self) -> CostEstimate;
}

pub trait Executable: Send + Sync {
    async fn exec(&self, ctx: Context) -> Result<Output, KairoError>;
}

pub trait Provider: Send + Sync {
    fn complete<'a>(
        &'a self,
        messages: Vec<Message>,
        options: CompletionOptions,
    ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, KairoError>> + Send + 'a>>;
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn execute<'a>(
        &'a self,
        input: ToolInput,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, KairoError>> + Send + 'a>>;
}

pub trait Connector: Send + Sync {
    async fn invoke(&self, action: Action) -> Result<Data, KairoError>;
}
