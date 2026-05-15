# SPINE Module Map (compact reference for TASKS phase)

## Key directories
- `spine/workflow/` — LangGraph StateGraph topology (compose.py, registry.py, critic_review.py, artifact_gate.py)
- `spine/phases/` — Per-phase node functions (specify.py, plan.py, tasks.py, implement.py, verify.py, critic.py)
- `spine/agents/` — Phase agent builders + factory (factory.py, helpers.py, profile.py, subagents.py, artifacts.py)
- `spine/work/` — Dispatcher (entry point), RalphLoopWorker (queue processor)
- `spine/models/` — WorkflowState TypedDict, enums, dataclasses
- `spine/persistence/` — CheckpointStore (SQLite), ArtifactStore (file-based)
- `spine/services/` — AuditService
- `spine/ui_api/` — UIApi (sole read/write interface for Streamlit)
- `spine/ui/` — Streamlit pages, WebSocket bus, components
- `spine/cli/` — Click commands (run, status, list, resume, worker, ui)
- `spine/config.py` — SpineConfig (loads .spine/config.yaml + .env)

## Artifact paths
- Work-item scoped: `.spine/artifacts/{work_id}/{phase}/{name}`
- Agent-readable: `.spine/artifacts/{phase}/{name}` (legacy, no work_id)

## Provider resolution
- Providers from `config["configurable"]["providers"]`, NEVER from WorkflowState
- Model resolution: `SpineConfig.resolve_model()` → `SPINE_MODEL` env → error
