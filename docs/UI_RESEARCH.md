# SPINE UI Research & Design

## Executive Summary

This document presents the research and design decisions for a web-based dashboard UI for the SPINE harness. The CLI (`spine work "..."`) is excellent for automation but lacks the visual debugging, real-time observability, and human-in-the-loop review capabilities needed for productive daily use of a multi-agent AI orchestration system.

---

## 1. Problem Statement

The SPINE harness is a state machine-driven AI agent orchestration system with:
- **State machine phases**: INIT → PLANNING → EXECUTION → VERIFICATION → COMPLETE
- **Parallel DAG execution**: Multiple agents working concurrently within phases
- **Swarm gates**: Critic review, pre-check batches, completion verification
- **Hive task tracking**: Durable task records with file reservations
- **Provider management**: LLM, memory, tools, storage, notifications
- **Persistence**: Five-layer model with checkpoint/restore

Currently, users interact via CLI. They can see phase progress (`rich` tables, console output) but cannot:
- Visualize the state machine and DAG in real-time
- See which agents are doing what, and their outputs
- Review critic gate decisions before execution proceeds
- Compare provider confidence scores visually
- Browse checkpoint history and resume from specific checkpoints
- Manage provider configurations with a form
- Inspect swarm mail events and file reservations

---

## 2. Requirements

### 2.1 Primary User Stories

1. **As a user**, I want to start a new work item from a web form and see real-time progress, so I can monitor multi-agent execution without staring at a terminal.
2. **As a reviewer**, I want to see the critic gate output and approve/reject plans before execution starts, so I maintain quality control.
3. **As an operator**, I want to see all running work items, their phase status, and agent activity at a glance, so I can manage concurrent workflows.
4. **As a developer**, I want to manage provider configurations (LLM, memory, storage) via a form, so I can switch models without editing YAML.
5. **As a user**, I want to browse checkpoint history and resume from any point, so I can recover interrupted work gracefully.

### 2.2 Non-Functional Requirements

- **Real-time updates**: WebSocket or polling for live progress during execution
- **Low barrier to setup**: Must work with minimal configuration — no separate frontend server needed
- **Python-native**: No separate Node.js build step, no bundlers, no npm
- **Portable**: Single `pip install` + `spine ui` command
- **Responsive**: Works on laptop screens, mobile-friendly

---

## 3. Technology Options Evaluated

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Streamlit** | Python-native, rapid development, built-in real-time, no build step, rich components (tables, charts, forms) | Limited layout control, not ideal for complex interactive apps | ✅\*..\. **RECOMMENDED** for v1 |
| **Gradio** | Simple forms, built for ML demos | Limited customization, not suited for dashboard-style UI | Reject |
| **Dash (Plotly)** | Powerful, reactive, good charts | Steep learning curve, verbose, heavy dependencies | Reject for v1 |
| **FastAPI + React** | Full control, production-grade, scalable | Requires build step, separate frontend, more ops burden | V2 target |
| **Reflex** | Python → React, type-safe | Newer ecosystem, fewer components, growing pains | Monitor |
| **NiceGUI** | Python-native, uses Vue.js, flexible layouts | Smaller community, fewer pre-built components | Monitor |

### Decision: Streamlit for v1

Streamlit is the pragmatic choice for an MVP because:
1. The entire SPINE codebase is Python + Rich CLI
2. We can build a full dashboard in a single day
3. Streamlit's `st.progress`, `st.status`, `st.tabs`, `st.dataframe` cover all needs
4. It can run alongside the CLI (same process, different entry point)
5. Later, we can extract components to a FastAPI + React frontend without rewriting logic

**Path to v2**: When the dashboard grows beyond Streamlit's capabilities, use the existing provider/backend APIs (which are already structured) and build a proper SPA frontend.

---

## 4. Proposed Architecture

```
┌─────────────────────────────────────────────────────┐
│                    SPINE UI                          │
│                   (Streamlit)                        │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Dashboard │  │Work Item │  │Providers │          │
│  │  (list)   │  │  Detail  │  │  Config  │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │State      │  │Swarm     │  │History   │          │
│  │Machine    │  │Mail /    │  │/Checkpts │          │
│  │Visualizer │  │Events    │  │          │          │
│  └──────────┘  └──────────┘  └──────────┘          │
├─────────────────────────────────────────────────────┤
│              Session State (st.session_state)        │
├─────────────────────────────────────────────────────┤
│           SPINE Core Backend (Python)                │
│  StateMachine │ DAG Executor │ Hive │ Providers      │
├─────────────────────────────────────────────────────┤
│           Persistence Layer (.spine/)                │
│  checkpoints/ │ events/ │ hive/ │ knowledge/         │
└─────────────────────────────────────────────────────┘
```

### Communication Model

Two approaches, both viable:

**A. Polling (simpler, recommended for v1):**
- Streamlit app polls the checkpoint database (SQLite) every 2-3 seconds
- Each work item's state is read from the last checkpoint
- Simple, reliable, no WebSocket setup needed

**B. WebSocket push (future):**
- SPINE backend pushes events via WebSocket
- UI receives real-time updates without polling
- Better for long-running workflows with many agents

---

## 5. UI Wireframes & Layout

### 5.1 Main Page — Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│  SPINE  ─────────────  New Work  │  Work Items  │  Providers  │  ⚙  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Active Work Items                              [Filter ▼]   │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │  🔵  Build auth system        PLANNING    ████████░░ 80%      │  │
│  │  🟡  Fix login bug            EXECUTION   ████░░░░░░ 40%      │  │
│  │  🟢  Add tests               COMPLETE     ██████████ 100%      │  │
│  │  🔴  Deploy to staging       BLOCKED      ░░░░░░░░░░  0%       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Interactions:**
- Click any work item → navigate to Detail view
- "New Work" opens the work creation form
- Click "Providers" → provider config page
- Click "⚙" → settings page

### 5.2 Work Creation Form

```
┌─────────────────────────────────────────────────────────────────────┐
│  ← Back    New Work Item                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Working Title:  [__________________________________________]       │
│                                                                     │
│  Description:                                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                                                              │  │
│  │                                                              │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Method:                                                              │  │
│  ○ Full Spec Driven (SDD)      ● Quick Work                        │
│                                                                     │
│  Project Type:                                                   │  │
│  ○ Greenfield                 ● Brownfield                        │
│                                                                     │
│  LLM Provider:    [qwen3:32b (Ollama)        ▼]                   │
│                                                                     │
│  [ Start Work → ]                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 Work Item Detail — Phase View

```
┌─────────────────────────────────────────────────────────────────────┐
│  ← Back    Build auth system                              [Resume]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Status: PLANNING (in progress)   Started: 2 min ago                │
│  Requirement: "Build a JWT auth middleware with RS256"              │
│                                                                     │
│  ── State Machine ──────────────────────────────────────────────    │
│                                                                     │
│  INIT ─→ PLANNING ● ─→ EXECUTION ○ ─→ VERIFICATION ○ ─→ COMPLETE ○ │
│             ↑           │                                          │
│         (current)      [Critic Gate Pending...]                     │
│                                                                     │
│  ── Current Phase: PLANNING ──────────────────────────────────    │
│                                                                     │
│  Sub-Phases:                                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 🟢 ANALYZE        (explorer)     ✓ Complete                    │  │
│  │ 🟢 RESEARCH       (sme)          ✓ Complete                    │  │
│  │ 🟡 SYNTHESIZE     (planner)      In progress... 65%            │  │
│  │ ⚪ CRITIC REVIEW  (critic)       Pending                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ── Agent Outputs ──────────────────────────────────────────────    │
│                                                                     │
│  [ Tabs: ANALYZE │ RESEARCH │ SYNTHESIZE │ CRITIC GATE ]           │
│                                                                     │
│  SYNTHESIZE ───────────────────────────────────────────────────     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Plan draft (generated by planner agent):                      │  │
│  │                                                               │  │
│  │ 1. Design JWT middleware architecture                         │  │
│  │ 2. Implement token generation with RS256 keys                 │  │
│  │ 3. Add token validation middleware                            │  │
│  │ 4. Write unit tests for each component                       │  │
│  │ 5. Integration test with mock OAuth2 provider                │  │
│  │                                                               │  │
│  │ Estimated: 4 tasks, ~2 hours                                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ── Swarm Events ──────────────────────────────────────────────    │
│  [ Timestamp │ From │ To │ Subject │ Preview... ]                   │
│  [ 10:32:01 │ planner │ critic  │ PLAN_FOR_REVIEW  │ ...          │  │
│  [ 10:30:45 │ sme     │ planner │ RESEARCH_FINDINGS  │ ...        │  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Critic Gate Review (Human-in-the-Loop)

```
┌─────────────────────────────────────────────────────────────────────┐
│  ← Back    Critic Gate Review — Build auth system                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ⚠️  The critic gate is blocking the transition to EXECUTION.       │
│  Please review the plan before it proceeds.                         │
│                                                                     │
│  ── Full Plan ──────────────────────────────────────────────────    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ ## Plan: JWT Auth Middleware                                  │  │
│  │                                                               │  │
│  ### Tasks                                                        │  │
│  1. Design architecture (analysis)                                │  │
│  2. Implement key generation (RS256, 2048-bit)                   │  │
│  3. Implement token signing/verification                          │  │
│  4. Implement middleware for request validation                    │  │
│  5. Unit tests for each component                                 │  │
│  6. Integration test with mock provider                           │  │
│  │                                                               │  │
│  ### Risks Identified by Critic                                   │  │
│  ⚠ Key storage should use env vars, not hardcoded                │  │
│  ⚠ Consider token refresh flow                                   │  │
│  ⚠ Rate limiting should be added for auth endpoints              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ── Critic Feedback ────────────────────────────────────────────    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ - Key management needs more detail                            │  │
│  │ - No refresh token flow described                             │
│  │ - Rate limiting is critical for auth, should be Task 7        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  [ ✓ Approve Plan & Continue → ]  [ ✎ Request Revisions ]          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.5 Provider Configuration

```
┌─────────────────────────────────────────────────────────────────────┐
│  ← Back    Provider Configuration                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Tabs: [ LLM Providers │ Memory │ Tools │ Storage │ Notify ]       │
│                                                                     │
│  ── LLM Providers ──────────────────────────────────────────────    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ + Add Provider                                                │  │
│  │                                                               │  │
│  │ Name: [ primary           ]   Type: [ openai ▼    ]            │  │
│  │ Model: [ gpt-4.1          ]   API Key: [••••••••••••  Show]   │  │
│  │ Base URL: [                      ]   Priority: [ 1 ]          │  │
│  │                                                               │  │
│  │ Name: [ local             ]   Type: [ ollama   ▼ ]            │  │
│  │ Model: [ qwen3:32b        ]   Base URL: [ http://localhost    │  │
│  │                                           :11434    ]          │  │
│  │ Priority: [ 2 ]                                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  [ Save Configuration → ]                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Navigation Map

```
Dashboard (list of work items)
  │
  ├── New Work Item
  │     └── Confirm → Starts execution → Redirect to Detail
  │
  ├── Work Item Detail
  │     ├── Phase View (default)
  │     │     ├── State Machine Visualizer
  │     │     ├── Sub-Phase Progress
  │     │     ├── Agent Output Viewer (tabbed by sub-phase)
  │     │     └── Swarm Events Log
  │     │
  │     ├── Critic Gate (if blocked)
  │     │     ├── View full plan
  │     │     ├── View critic feedback
  │     │     └── Approve / Request Revisions
  │     │
  │     ├── Hive Tasks
  │     │     ├── Task list with status
  │     │     ├── File reservations
  │     │     └── Task detail
  │     │
  │     ├── Artifacts
  │     │     ├── Generated plans
  │     │     ├── Solutions
  │     │     └── Verification reports
  │     │
  │     └── History
  │           └── Checkpoint list with timestamps
  │
  ├── Provider Configuration
  │     ├── LLM providers tab
  │     ├── Memory tab
  │     ├── Tools tab
  │     ├── Storage tab
  │     └── Notify tab
  │
  └── Settings
        ├── Theme (light/dark)
        ├── Auto-refresh interval
        └── Reset all state
```

---

## 7. Technical Implementation Plan

### 7.1 File Structure

```
spine/
├── ui/                          # NEW: Streamlit dashboard
│   ├── __init__.py
│   ├── app.py                   # Main Streamlit entry point
│   ├── dashboard.py             # Dashboard page (work items list)
│   ├── work_detail.py           # Work item detail view
│   ├── work_new.py              # Work creation form
│   ├── providers_config.py      # Provider configuration UI
│   ├── state_machine_viz.py     # State machine visualizer component
│   ├── critic_gate.py           # Critic gate review component
│   ├── components.py            # Shared components (progress bars, status badges)
│   └── utils.py                 # Shared helpers (checkpoint loading, etc.)
│
├── core/
│   ├── ui_api.py                # NEW: Clean API for UI to read/write state
│   │                            #   - get_active_work_items()
│   │                            #   - get_work_item_detail(id)
│   │                            #   - start_work(requirement)
│   │                            #   - approve_gate(work_item_id)
│   │                            #   - reject_gate(work_item_id, feedback)
│   │                            #   - get_checkpoints(work_item_id)
│   │                            #   - resume_from_checkpoint(checkpoint_id)
│   │                            #   - update_provider_config(config)
│   │                            #   - get_provider_config()
```

### 7.2 Key Design Decisions

1. **Single-process architecture**: Streamlit runs in the same process as the SPINE backend. No separate frontend server.
2. **Session state management**: Use `st.session_state` for navigation state, filter state, and expanded sections.
3. **Real-time via polling**: Poll checkpoint DB every 2-3 seconds. For long work items, this is sufficient. WebSocket is a future optimization.
4. **Read-only by default**: Most UI pages are read-only. Write operations (approve, reject, start work) go through explicit buttons.
5. **Thread safety**: The Streamlit app reads from the SQLite checkpoint DB (which LangGraph uses). Reads are safe during writes. For writing (approving gates), use file-based locks.
6. **Backwards compatible**: The CLI continues to work. The UI is an optional entry point.

### 7.3 Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    ...existing...
    "streamlit>=1.30.0",
    "plotly>=5.18.0",        # For DAG visualization
]
```

Optional v2 dependencies (not for v1):
```toml
[project.optional-dependencies]
ui = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "websockets>=12.0",
]
```

---

## 8. Future Enhancements (Post-v1)

### 8.1 Near-Term (v1.1 - v1.2)

1. **DAG visualization**: Interactive graph of the execution DAG with node colors indicating status (plotly or graphviz)
2. **WebSocket real-time**: Push agent events to the UI without polling
3. **Dark mode**: Streamlit supports this natively
4. **Agent output streaming**: Show LLM agent responses as they stream in
5. **Search/filter work items**: By phase, status, provider, date

### 8.2 Medium-Term (v2)

1. **SPA frontend**: FastAPI + React/Next.js with proper routing
2. **Multi-repo support**: Dashboard for monitoring SPINE across multiple repos
3. **Provider benchmarking**: Compare LLM provider performance side-by-side
4. **Learning dashboard**: View pattern success rates, anti-patterns
5. **Mobile app**: React Native or PWA for on-the-go monitoring
6. **Plugin marketplace**: Browse and install provider plugins

### 8.3 Long-Term (v3)

1. **Collaborative review**: Multiple humans reviewing gates simultaneously
2. **Audit trail viewer**: Full event timeline for any work item
3. **Simulator**: Test workflows with synthetic data before running
4. **API access**: REST API for all UI operations (CLI + UI are just clients)

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Streamlit session state bugs | Medium | Use explicit checkpoint DB reads instead of session state caching |
| Long work items block UI | Low | Streamlit auto-re-renders; polling handles async execution |
| Thread safety with concurrent CLI + UI | Medium | All reads are from SQLite (thread-safe reads); writes use file locks |
| Provider config conflicts (CLI vs UI) | Low | Both read/write same `.spine/config.yaml`; UI adds validation |
| Large checkpoint payloads slow rendering | Low | Paginate checkpoint history; lazy-load agent outputs |
| Streamlit can't handle complex layouts | Medium | If it becomes a problem, graduate to FastAPI+React (documented path) |

---

## 10. Summary of Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | **Streamlit** | Python-native, rapid dev, fits SPINE's stack |
| Architecture | **Single-process** | No extra services, simple deployment |
| Real-time | **Polling (2-3s)** | Simple, reliable, sufficient for v1 |
| Write safety | **Explicit buttons** | No auto-save, human must confirm actions |
| Data access | **Read checkpoint DB directly** | Works with LangGraph's SQLiteSaver as-is |
| CLI compatibility | **Both work independently** | CLI and UI share backend, no conflicts |
| Next UI version | **FastAPI + React** | When Streamlit becomes limiting |
