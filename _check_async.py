#!/usr/bin/env python3
"""Quick verification that all phase call_fns are async."""
import asyncio
from spine.workflow.registry import get_registry

registry = get_registry()
for name, phase_def in registry.all_phases().items():
    fn = phase_def.call_fn
    is_async = asyncio.iscoroutinefunction(fn)
    print(f'{name}: call_fn={fn.__name__}, async={is_async}')
