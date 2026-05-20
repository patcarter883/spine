from spine.agents.interpreter import build_interpreter_middleware

code = """
class SpineCodeInterpreterMiddleware(CodeInterpreterMiddleware):
    async def before_model(
        self,
        messages: list[Any],
        *,
        tools: list[Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = await super().before_model(messages, tools=tools, **kwargs)
        
        # The prompt is injected during before_model into a SystemMessage or the first message
        # Let's search the messages and strip the API Reference
        new_msgs = result.get("messages", messages)
        import re
        
        for i, m in enumerate(new_msgs):
            if hasattr(m, 'content') and isinstance(m.content, str):
                # Rip out the entire block starting from "### API Reference — `tools` namespace"
                m.content = re.sub(
                    r"### API Reference — `tools` namespace.*?```typescript.*?```",
                    "",
                    m.content,
                    flags=re.DOTALL
                )
        return result
"""
