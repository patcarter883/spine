#!/usr/bin/env python
"""Quick test for dashboard showing completed tasks."""

import sys
sys.path.insert(0, '/home/pat/projects/spine')

from spine.ui.utils import get_active_work_items
from spine.cli.commands.status import get_threads

# Test get_threads from CLI
threads = get_threads('.spine/spine.db')
print(f'CLI get_threads found {len(threads)} threads:')
for t in threads:
    print(f'  - {t["thread_id"][:8]}... | phase={t["phase"]} | req={t.get("requirement", "")[:40]}')

print()

# Test get_active_work_items from UI
items = get_active_work_items()
print(f'UI get_active_work_items found {len(items)} items:')
for item in items:
    print(f'  - {item["thread_id"][:8]}... | phase={item["phase"]} | status={item["status"]}')

print()

# Count by phase
active = sum(1 for w in items if w["phase"] not in ("COMPLETE", "ERROR", "BLOCKED"))
complete = sum(1 for w in items if w["phase"] == "COMPLETE")
errors = sum(1 for w in items if w["phase"] == "ERROR")

print(f'Summary: {active} active, {complete} complete, {errors} errors')
