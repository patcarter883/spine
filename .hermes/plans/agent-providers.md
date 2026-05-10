# Plan: Agent Provider Integration for SPINE

## Goal
Add an `AgentProvider` layer to SPINE that lets swarm agents delegate work to
external coding agents (OpenCode, Codex CLI, Claude Code SDK) via a unified
interface. The default is OpenCode (provider-agnostic). Codex and Claude Code
are optional. All are configured via `spine.yaml`.

## Architecture

```
SwarmAgent (coder, test_engineer, reviewer, ...)
    │
    ▼
AgentProvider (new ProviderType)
    │
    ├── OpenCodeAgentProvider   ← PRIMARY (any LLM provider)
    │     opencode run "..." —model provider/model
    │     opencode serve → HTTP API
    │     opencode acp  → JSON-RPC stdio
    │
    ├── CodexAgentProvider      ← OPTIONAL (OpenAI only)
    │     codex exec "..." --sandbox workspace-write
    │
    └── ClaudeCodeAgentProvider ← OPTIONAL (Anthropic only)
          claude-agent-sdk (Python)
```

## Files to Create

1. **spine/providers/agents.py** — AgentProvider base class + all implementations
2. **tests/test_agent_providers.py** — Tests for the agent provider layer

## Files to Modify

3. **spine/providers/base.py** — Add `AGENT = "agent"` to ProviderType enum
4. **spine/providers/__init__.py** — Export new agent provider classes
5. **spine/swarm/agents.py** — Add agent_provider to SwarmAgent
6. **spine/models/types.py** — Add `agent_provider` to SpineState TypedDict
7. **pyproject.toml** — Add `httpx` dependency (for OpenCode HTTP API)

## Implementation Steps

### Step 1: Add AGENT to ProviderType (base.py)
One-line addition to the enum.

### Step 2: Create AgentProvider base class (agents.py)
- `AgentProvider(Provider)` with `provider_type = ProviderType.AGENT`
- Abstract method: `execute(prompt, workdir, **kwargs) -> AgentResult`
- Abstract method: `is_available() -> bool`
- `AgentResult` dataclass: output, exit_code, files_changed, error, metadata

### Step 3: Create OpenCodeAgentProvider (agents.py)
Three integration modes:
- `run` mode: subprocess `opencode run "prompt" --model X --json`
- `serve` mode: HTTP POST to `opencode serve` API
- `acp` mode: JSON-RPC over stdio via `opencode acp`

Default mode: `run` (simplest, no server needed).

### Step 4: Create CodexAgentProvider (agents.py)
- Wraps `codex exec "prompt" --sandbox workspace-write`
- OpenAI-only, optional dependency

### Step 5: Create ClaudeCodeAgentProvider (agents.py)
- Uses `claude-agent-sdk` Python package if installed
- Falls back to `claude` CLI subprocess
- Anthropic-only, optional dependency

### Step 6: Wire into SwarmAgent (swarm/agents.py)
- Add `agent_provider` parameter to SwarmAgent
- When agent_provider is set, `execute()` delegates to it for implementation tasks
- When only llm_provider is set, behavior unchanged (for decision-making agents)

### Step 7: Add to SpineState (models/types.py)
- Add `agent_provider: Optional[dict[str, Any]]` to SpineState TypedDict

### Step 8: Update providers/__init__.py
- Export all new classes

### Step 9: Update pyproject.toml
- Add `httpx>=0.27.0` as dependency (for OpenCode HTTP API)

### Step 10: Write tests (tests/test_agent_providers.py)
- Test AgentResult dataclass
- Test OpenCodeAgentProvider.build_command()
- Test CodexAgentProvider.build_command()
- Test AgentFallbackChain
- Test unavailable providers (no opencode installed)