"""将 LangGraph 节点接入权限、超时和重试控制的工具网关。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from paper_agent.security import ActorContext, ToolAuthorizationPolicy
from paper_agent.tools.orchestrator import ResourceAccess, ToolCall, ToolOrchestrator, ToolSpec, ToolStatus


@dataclass(frozen=True)
class ToolRuntimePolicy:
    """受控工作流节点的超时、资源和重试配置。"""

    timeout_s: float = 120.0
    max_retries: int = 0
    resources: frozenset[str] = field(default_factory=lambda: frozenset({"paper_qa"}))
    access: ResourceAccess = ResourceAccess.READ


class AuthorizedToolGateway:
    """将确定性节点封装成受授权、受超时保护的工具调用。

    这里只接收代码闭包，LLM 无法传入工具名、Scope 或 Python 表达式。
    """

    def __init__(
        self,
        actor: ActorContext,
        *,
        authorizer: ToolAuthorizationPolicy | None = None,
        policies: dict[str, ToolRuntimePolicy] | None = None,
    ) -> None:
        self.actor = actor
        self.authorizer = authorizer or ToolAuthorizationPolicy()
        self.policies = policies or {}

    def run(self, name: str, callback: Callable[[], Any]) -> Any:
        """执行一个允许列表内节点；失败保留明确错误供 LangGraph 安全终止。"""

        self.authorizer.authorize(self.actor, name)
        policy = self.policies.get(name, ToolRuntimePolicy())
        orchestrator = ToolOrchestrator(
            [
                ToolSpec(
                    name=name,
                    handler=lambda _args: callback(),
                    timeout_s=policy.timeout_s,
                    max_retries=policy.max_retries,
                    resources=policy.resources,
                    access=policy.access,
                    input_schema="{}",
                    output_schema="runtime-defined",
                    required_scopes=self.authorizer.DEFAULT_SCOPES[name],
                )
            ]
        )
        result = orchestrator.run_many([ToolCall(call_id=f"node:{name}", tool_name=name)])[0]
        if result.status == ToolStatus.SUCCESS:
            return result.output
        raise RuntimeError(f"工具 {name} 执行失败：{result.status.value}；{result.error or '未知错误'}")
