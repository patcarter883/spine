"""SPINE Projects page — list, view, create, and manage projects.

A project is a persistent envelope grouping many top-level work items
(``member_work_ids`` is the source of truth). This single page renders either
the project list or, when ``?project_id=`` is set, a project detail view with
its deterministic requirement-coverage rollup, the work items grouped under it,
and full management controls (create, edit the spec, add/remove members,
delete). All data access goes through :class:`UIApi`.
"""

from __future__ import annotations

import streamlit as st

from spine.ui.pages import get as get_page
from spine.ui.utils import status_icon, truncate
from spine.ui_api import UIApi

# ── Fragment refresh interval (seconds) ──
_POLL_INTERVAL = 10

# Status colours for the coverage rollup, matching the CLI `project show`.
_COVERAGE_COLORS = {
    "satisfied": "green",
    "partial": "orange",
    "unsatisfied": "red",
}
_PHASE_COLORS = {
    "complete": "green",
    "in_progress": "orange",
    "pending": "gray",
}


# ── Spec-editor helpers (shared by create + edit) ──


def _lines_to_list(text: str) -> list[str]:
    """Split a one-per-line text area into a clean list of strings."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _requirements_editor(
    existing: list[dict] | None, *, key: str, lock_ids: bool
) -> list[dict]:
    """Render an editable requirements table and return the marshalled rows.

    ``lock_ids`` keeps the ``id`` column read-only on edit (requirement IDs are
    immutable — the coverage aggregator keys off them).
    """
    rows = [
        {
            "id": r.get("id", ""),
            "text": r.get("text", ""),
            "rationale": r.get("rationale", ""),
        }
        for r in (existing or [])
    ]
    edited = st.data_editor(
        rows,
        key=key,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn(
                "ID", help="Stable, immutable (e.g. R-001).", disabled=lock_ids
            ),
            "text": st.column_config.TextColumn("Requirement", width="large"),
            "rationale": st.column_config.TextColumn("Rationale"),
        },
    )
    out: list[dict] = []
    for i, row in enumerate(edited):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        rid = (row.get("id") or "").strip() or f"R-{i + 1:03d}"
        out.append({"id": rid, "text": text, "rationale": (row.get("rationale") or "").strip()})
    return out


def _phases_editor(
    existing: list[dict] | None, *, key: str, lock_ids: bool
) -> list[dict]:
    """Render an editable roadmap-phases table and return the marshalled rows.

    ``requirement_ids`` / ``member_work_ids`` are entered as comma-separated
    strings and parsed back into lists.
    """
    rows = [
        {
            "id": p.get("id", ""),
            "title": p.get("title", ""),
            "description": p.get("description", ""),
            "requirement_ids": ", ".join(p.get("requirement_ids", [])),
            "member_work_ids": ", ".join(p.get("member_work_ids", [])),
        }
        for p in (existing or [])
    ]
    edited = st.data_editor(
        rows,
        key=key,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn(
                "ID", help="Stable, immutable (e.g. M-001).", disabled=lock_ids
            ),
            "title": st.column_config.TextColumn("Title"),
            "description": st.column_config.TextColumn("Description", width="large"),
            "requirement_ids": st.column_config.TextColumn("Requirement IDs (comma-sep)"),
            "member_work_ids": st.column_config.TextColumn("Member work IDs (comma-sep)"),
        },
    )

    def _split(value: str) -> list[str]:
        return [v.strip() for v in (value or "").split(",") if v.strip()]

    out: list[dict] = []
    for i, row in enumerate(edited):
        title = (row.get("title") or "").strip()
        if not title:
            continue
        pid = (row.get("id") or "").strip() or f"M-{i + 1:03d}"
        out.append(
            {
                "id": pid,
                "title": title,
                "description": (row.get("description") or "").strip(),
                "requirement_ids": _split(row.get("requirement_ids", "")),
                "member_work_ids": _split(row.get("member_work_ids", "")),
            }
        )
    return out


# ── List view ──


@st.fragment(run_every=_POLL_INTERVAL)
def _render_project_list(api: UIApi) -> None:
    """Auto-refreshing table of all projects."""
    projects = api.list_projects()
    if not projects:
        st.info("No projects yet. Create one below.")
        return

    for proj in projects:
        col1, col2, col3, col4 = st.columns([3, 5, 1, 1])
        col1.markdown(f"**{proj['id']}**")
        col2.write(proj.get("title", ""))
        col3.write(f"👥 {proj.get('members', 0)}")
        if col4.button("View", key=f"view_{proj['id']}"):
            st.switch_page(get_page("projects"), query_params={"project_id": proj["id"]})


def _render_create_form(api: UIApi) -> None:
    """Full-spec create form inside an expander."""
    with st.expander("➕ Create project"):
        project_id = st.text_input("Project ID (slug)", key="create_pid")
        title = st.text_input("Title", key="create_title")
        summary = st.text_area("Summary", key="create_summary", height=80)
        objectives = st.text_area(
            "Objectives (one per line)", key="create_objectives", height=80
        )
        constraints = st.text_area(
            "Constraints (one per line)", key="create_constraints", height=80
        )
        hard_boundaries = st.text_area(
            "Hard boundaries (one per line)", key="create_boundaries", height=80
        )

        st.caption("Requirements")
        requirements = _requirements_editor(None, key="create_reqs", lock_ids=False)

        st.caption("Roadmap phases")
        phases = _phases_editor(None, key="create_phases", lock_ids=False)

        if st.button(
            "Create",
            type="primary",
            key="create_submit",
            disabled=not project_id.strip(),
        ):
            result = api.create_project(
                project_id=project_id.strip(),
                title=title.strip() or None,
                summary=summary.strip(),
                objectives=_lines_to_list(objectives),
                requirements=requirements,
                constraints=_lines_to_list(constraints),
                hard_boundaries=_lines_to_list(hard_boundaries),
                roadmap={"phases": phases},
            )
            if "error" in result:
                st.error(f"Failed: {result['error']}")
            else:
                st.success(f"Created project '{result['id']}'.")
                st.switch_page(
                    get_page("projects"), query_params={"project_id": result["id"]}
                )


def _render_list_view(api: UIApi) -> None:
    st.title("📁 Projects")
    st.markdown(
        "Projects group many work items under a shared spec and roadmap. "
        "Coverage is computed from members that have run and passed verification."
    )
    _render_project_list(api)
    st.divider()
    _render_create_form(api)


# ── Detail view ──


@st.fragment(run_every=_POLL_INTERVAL)
def _render_coverage(api: UIApi, project_id: str) -> None:
    """Auto-refreshing requirement-coverage rollup."""
    coverage = api.get_project_coverage(project_id)
    if coverage is None:
        st.warning("Coverage unavailable.")
        return

    summary = coverage["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Members", coverage["total_members"])
    c2.metric("Satisfied", summary["satisfied"])
    c3.metric("Partial", summary["partial"])
    c4.metric("Unsatisfied", summary["unsatisfied"])
    st.caption(
        f"{coverage['members_with_state']} with state · "
        f"{coverage['verified_members']} verified"
    )

    if coverage["requirements"]:
        st.subheader("Requirement coverage")
        for r in coverage["requirements"]:
            color = _COVERAGE_COLORS.get(r["status"], "gray")
            st.markdown(
                f"- **{r['id']}** :{color}[{r['status']}] "
                f"({len(r['verified'])}/{len(r['covering'])}) — {truncate(r['text'], 80)}"
            )

    if coverage["phases"]:
        st.subheader("Roadmap phases")
        for p in coverage["phases"]:
            color = _PHASE_COLORS.get(p["status"], "gray")
            st.markdown(f"- :{color}[{p['status']}] — **{p['id']}**: {p['title']}")


@st.fragment(run_every=_POLL_INTERVAL)
def _render_members(api: UIApi, project_id: str) -> None:
    """Auto-refreshing list of member work items, grouped under the project."""
    members = api.get_project_members(project_id)
    st.subheader(f"Work items ({len(members)})")
    if not members:
        st.info("No work items belong to this project yet.")
        return
    for m in members:
        col1, col2, col3 = st.columns([2, 5, 1])
        col1.markdown(f"`{m['work_id']}`")
        phase = f" · {m['current_phase']}" if m.get("current_phase") else ""
        col2.write(
            f"{status_icon(m['status'])} {m['status']}{phase} — "
            f"{truncate(m['description'], 60)}"
        )
        if col3.button("Open", key=f"open_{m['work_id']}"):
            st.switch_page(
                get_page("work-detail"), query_params={"work_id": m["work_id"]}
            )


def _render_management(api: UIApi, project_id: str, spec: dict) -> None:
    """Add/remove members, edit the spec, and delete the project."""
    members = list(spec.get("member_work_ids", []))

    with st.expander("➕ Add members"):
        all_work = api.list_work(limit=200)
        candidates = [w["id"] for w in all_work if w.get("id") and w["id"] not in members]
        if not candidates:
            st.caption("No unassigned work items available.")
        else:
            to_add = st.multiselect("Work items to add", candidates, key="add_members")
            if st.button("Add", key="add_members_btn", disabled=not to_add):
                result = api.add_project_members(project_id, to_add)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Added {len(to_add)} member(s).")
                    st.rerun()

    with st.expander("➖ Remove members"):
        if not members:
            st.caption("No members to remove.")
        else:
            to_remove = st.multiselect("Members to remove", members, key="remove_members")
            if st.button("Remove", key="remove_members_btn", disabled=not to_remove):
                result = api.remove_project_members(project_id, to_remove)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Removed {len(to_remove)} member(s).")
                    st.rerun()

    with st.expander("✏️ Edit spec"):
        title = st.text_input("Title", value=spec.get("title", ""), key="edit_title")
        summary = st.text_area(
            "Summary", value=spec.get("summary", ""), key="edit_summary", height=80
        )
        objectives = st.text_area(
            "Objectives (one per line)",
            value="\n".join(spec.get("objectives", [])),
            key="edit_objectives",
            height=80,
        )
        constraints = st.text_area(
            "Constraints (one per line)",
            value="\n".join(spec.get("constraints", [])),
            key="edit_constraints",
            height=80,
        )
        hard_boundaries = st.text_area(
            "Hard boundaries (one per line)",
            value="\n".join(spec.get("hard_boundaries", [])),
            key="edit_boundaries",
            height=80,
        )

        st.caption("Requirements (IDs are immutable)")
        requirements = _requirements_editor(
            spec.get("requirements"), key="edit_reqs", lock_ids=True
        )

        st.caption("Roadmap phases (IDs are immutable)")
        phases = _phases_editor(
            spec.get("roadmap", {}).get("phases"), key="edit_phases", lock_ids=True
        )

        if st.button("Save changes", type="primary", key="edit_submit"):
            result = api.update_project(
                project_id,
                title=title.strip() or project_id,
                summary=summary.strip(),
                objectives=_lines_to_list(objectives),
                requirements=requirements,
                constraints=_lines_to_list(constraints),
                hard_boundaries=_lines_to_list(hard_boundaries),
                roadmap={"phases": phases},
            )
            if "error" in result:
                st.error(f"Failed: {result['error']}")
            else:
                st.success("Project updated.")
                st.rerun()

    with st.expander("🗑️ Delete project"):
        st.warning("This permanently removes the project spec. Members are not deleted.")
        confirm = st.text_input(
            f"Type '{project_id}' to confirm", key="delete_confirm"
        )
        if st.button(
            "Delete project",
            type="primary",
            key="delete_btn",
            disabled=confirm != project_id,
        ):
            result = api.delete_project(project_id)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Project deleted.")
                st.switch_page(get_page("projects"), query_params={})


def _render_detail_view(api: UIApi, project_id: str) -> None:
    spec = api.get_project(project_id)
    if spec is None:
        st.error(f"Project '{project_id}' not found.")
        if st.button("← Back to projects"):
            st.switch_page(get_page("projects"), query_params={})
        return

    if st.button("← Back to projects"):
        st.switch_page(get_page("projects"), query_params={})

    st.title(f"📁 {spec.get('title', project_id)}")
    st.caption(f"ID: `{project_id}`")
    if spec.get("summary"):
        st.write(spec["summary"])
    if spec.get("objectives"):
        st.markdown("**Objectives:**")
        for obj in spec["objectives"]:
            st.markdown(f"- {obj}")

    st.divider()
    _render_coverage(api, project_id)

    st.divider()
    _render_members(api, project_id)

    st.divider()
    st.subheader("Manage")
    _render_management(api, project_id, spec)


# ── Entry point ──


def render(api: UIApi) -> None:
    """Render the projects list, or a project detail view when ``project_id`` is set."""
    project_id = st.query_params.get("project_id", "")
    if project_id:
        _render_detail_view(api, project_id)
    else:
        _render_list_view(api)
