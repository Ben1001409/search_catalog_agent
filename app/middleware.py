from __future__ import annotations

from typing import Any, Callable, Dict

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from state import AgentState


class AgentLoggingMiddleware(AgentMiddleware[AgentState]):
    state_schema = AgentState

    def before_model(self, state: AgentState, runtime) -> Dict[str, Any] | None:
        
        messages = state.get("messages", [])
        estimated_tokens = sum(len(str(getattr(m, "content", m))) for m in messages) // 4
        return {"total_estimated_tokens": estimated_tokens}

    def wrap_tool_call(self, request, handler):
        tool_name = getattr(request.tool_call, "name", None) or request.tool_call.get("name", "unknown")
        existing = list(request.state.get("tool_log", []))
        existing.append({"tool": tool_name})
        request = request.override(state={**request.state, "tool_log": existing})
        return handler(request)
