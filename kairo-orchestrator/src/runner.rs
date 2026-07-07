use async_trait::async_trait;
use kairo_agents::AgentResult;
use kairo_core::context::Context;
use kairo_core::error::KairoError;

#[async_trait]
pub trait TaskRunner: Send + Sync {
    async fn run(&self, ctx: Context) -> Result<AgentResult, KairoError>;
}

#[async_trait]
impl TaskRunner for kairo_agents::ReActAgent {
    async fn run(&self, ctx: Context) -> Result<AgentResult, KairoError> {
        self.run(ctx).await
    }
}

