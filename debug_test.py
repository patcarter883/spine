from unittest.mock import patch, MagicMock
from spine.agents.specify_agent import build_specify_agent
from spine.models.state import WorkflowState

def _make_state() -> WorkflowState:
    return {"work_id": "test-123"}

def _mock_create_agent(*args, **kwargs):
    return MagicMock()

@patch("spine.agents.factory.interpreter_enabled", return_value=True)
@patch("spine.agents.factory.create_agent")
@patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
def test(mock_or, mock_ca, mock_enabled):
    mock_ca.return_value = _mock_create_agent()
    build_specify_agent(_make_state(), None)
    call_kwargs = mock_ca.call_args[1]
    middleware = call_kwargs.get("middleware", [])
    print([type(m).__name__ for m in middleware])

test()
