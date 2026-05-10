"""Output format specifications for different capabilities.

Each format specifies the exact JSON structure agents should return.
This ensures consistency across the workflow and enables downstream
agents to parse results reliably.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from spine.prompts.base import Role


class OutputFormat(Enum):
    """Standard output format types."""
    ANALYSIS = "analysis"
    PLAN = "plan"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    RESEARCH = "research"
    TEST = "test"
    DEFAULT = "default"


# =============================================================================
# ANALYSIS OUTPUT FORMAT
# =============================================================================

ANALYSIS_OUTPUT_FORMAT = """## Output Format

Return a JSON object with this structure:

```json
{
  "problem_statement": "One clear sentence describing the core problem",
  "context": "Background information",
  "explicit_requirements": ["requirement1", "requirement2"],
  "implicit_requirements": ["requirement1"],
  "constraints": {
    "technical": ["constraint1"],
    "time": "deadline or 'none specified'",
    "resources": "resource constraints"
  },
  "success_criteria": ["criterion1", "criterion2"],
  "dependencies": {
    "prerequisites": ["what must exist first"],
    "integrations": ["systems to integrate with"]
  },
  "ambiguities": [
    {
      "question": "What is unclear?",
      "impact": "Why does this matter?",
      "suggested_resolution": "How might we resolve it?"
    }
  ],
  "suggested_phases": [
    {
      "name": "Phase name",
      "description": "What this phase accomplishes"
    }
  ],
  "risk_areas": [
    {
      "risk": "What could go wrong",
      "probability": "low|medium|high",
      "mitigation": "How to address it"
    }
  ]
}
```

**Important**: Always return valid JSON. Use empty arrays [] for missing optional fields.
"""

# =============================================================================
# PLAN OUTPUT FORMAT
# =============================================================================

PLAN_OUTPUT_FORMAT = """## Output Format

Return a JSON object with this structure:

```json
{
  "plan_summary": "One sentence describing the overall approach",
  "architecture": {
    "components": [
      {
        "name": "ComponentName",
        "purpose": "What this component does",
        "files": ["path/to/file.py"],
        "dependencies": ["OtherComponent"]
      }
    ],
    "data_flow": "Description of data flow"
  },
  "tasks": [
    {
      "id": "TASK-001",
      "title": "Task title",
      "description": "Detailed description",
      "type": "create|modify|delete|test|document",
      "files": ["path/to/file.py"],
      "dependencies": ["TASK-000"],
      "acceptance_criteria": ["criterion 1"],
      "estimated_complexity": "low|medium|high",
      "risk_level": "low|medium|high"
    }
  ],
  "testing_strategy": {
    "unit_tests": ["What to unit test"],
    "integration_tests": ["What to integration test"]
  },
  "checkpoints": [
    {
      "after_task": "TASK-XXX",
      "verify": ["What to verify"],
      "rollback_to": "TASK-XXX"
    }
  ],
  "assumptions": ["Assumption made"],
  "blocked_by": [
    {
      "blocker": "What is blocking",
      "resolution": "How to resolve"
    }
  ]
}
```

**Important**: Tasks must be ordered by dependency. Always return valid JSON.
"""

# =============================================================================
# IMPLEMENTATION OUTPUT FORMAT
# =============================================================================

IMPLEMENTATION_OUTPUT_FORMAT = """## Output Format

Return a JSON object with this structure:

```json
{
  "status": "success|partial|failed",
  "summary": "Brief description of what was implemented",
  "files_changed": [
    {
      "path": "path/to/file.py",
      "changes": "Description of changes",
      "lines_added": 10,
      "lines_removed": 5
    }
  ],
  "tests": {
    "added": ["test_file.py::test_name"],
    "passed": ["test_file.py::test_name"],
    "failed": [],
    "skipped": []
  },
  "acceptance_met": [
    {
      "criterion": "Description",
      "met": true,
      "evidence": "How verified"
    }
  ],
  "remaining_work": ["Work that couldn't be completed"],
  "notes": "Important notes or decisions",
  "rollback_instructions": "How to undo if needed"
}
```

**Important**: Always run tests before reporting success. Return valid JSON.
"""

# =============================================================================
# REVIEW OUTPUT FORMAT
# =============================================================================

REVIEW_OUTPUT_FORMAT = """## Output Format

Return a JSON object with this structure:

```json
{
  "status": "approved|changes_requested|rejected",
  "summary": "Overall assessment",
  "correctness": {
    "score": 8,
    "issues": [
      {
        "severity": "major",
        "description": "What is wrong",
        "location": "file.py:42",
        "suggestion": "How to fix"
      }
    ]
  },
  "security": {
    "score": 9,
    "vulnerabilities": []
  },
  "performance": {
    "score": 7,
    "concerns": [
      {
        "severity": "minor",
        "description": "Performance concern",
        "location": "file.py:100",
        "suggestion": "Improvement"
      }
    ]
  },
  "maintainability": {
    "score": 8,
    "issues": []
  },
  "completeness": {
    "requirements_met": ["req1"],
    "requirements_missing": [],
    "tests_adequate": true,
    "documentation_adequate": true
  },
  "required_changes": [
    {
      "priority": "blocker",
      "description": "What must change",
      "reason": "Why it must change"
    }
  ],
  "recommendations": [
    {
      "description": "Suggested improvement",
      "benefit": "What benefit it provides"
    }
  ]
}
```

**Important**: Scores are 1-10. Distinguish required changes from recommendations.
"""

# =============================================================================
# RESEARCH OUTPUT FORMAT
# =============================================================================

RESEARCH_OUTPUT_FORMAT = """## Output Format

Return a JSON object with this structure:

```json
{
  "research_question": "The question researched",
  "summary": "One-paragraph summary",
  "best_practices": [
    {
      "practice": "Practice name",
      "description": "What and why",
      "implementation": "How to apply",
      "references": ["Source"]
    }
  ],
  "existing_patterns": [
    {
      "location": "path/to/file.py",
      "pattern": "What pattern exists",
      "relevance": "How it applies"
    }
  ],
  "recommended_approach": {
    "approach": "Approach name",
    "rationale": "Why recommended",
    "steps": ["Step 1", "Step 2"],
    "alternatives_considered": [
      {
        "approach": "Alternative",
        "rejected_because": "Why not chosen"
      }
    ]
  },
  "pitfalls": [
    {
      "pitfall": "What to avoid",
      "why": "Why problematic",
      "prevention": "How to avoid"
    }
  ],
  "security_considerations": ["Consideration 1"],
  "performance_considerations": ["Consideration 1"],
  "libraries_and_tools": [
    {
      "name": "Library name",
      "purpose": "What it does",
      "recommendation": "use|avoid|neutral",
      "notes": "Additional context"
    }
  ],
  "confidence_level": "high|medium|low",
  "uncertainties": ["What remains uncertain"],
  "sources": [
    {
      "type": "documentation",
      "title": "Source title",
      "url": "URL if applicable",
      "relevance": "Why relevant"
    }
  ]
}
```

**Important**: Be honest about confidence level. Distinguish facts from opinions.
"""

# =============================================================================
# TEST OUTPUT FORMAT
# =============================================================================

TEST_OUTPUT_FORMAT = """## Output Format

Return a JSON object with this structure:

```json
{
  "summary": "What was tested",
  "test_files": [
    {
      "path": "tests/test_feature.py",
      "tests_added": ["test_name1", "test_name2"],
      "coverage": "estimated percentage"
    }
  ],
  "test_categories": {
    "unit": ["test_file.py::test_name"],
    "integration": ["test_file.py::test_name"],
    "edge_cases": ["test_file.py::test_name"]
  },
  "coverage_report": {
    "lines_covered": 100,
    "branches_covered": 95,
    "missing_coverage": ["file.py:42"]
  },
  "issues_found": [
    {
      "test": "test_name",
      "issue": "What the test revealed",
      "severity": "critical|major|minor"
    }
  ]
}
```

**Important**: Tests must be deterministic. Include edge cases.
"""

# =============================================================================
# DEFAULT OUTPUT FORMAT
# =============================================================================

DEFAULT_OUTPUT_FORMAT = """## Output Format

Return a JSON object with this structure:

```json
{
  "result": "The main result or output",
  "status": "success|partial|failed",
  "details": {
    "key": "value"
  },
  "notes": "Any additional notes",
  "next_steps": ["Suggested next steps"]
}
```

**Important**: Always return valid JSON. Use null for missing optional fields.
"""


# =============================================================================
# FORMAT REGISTRY
# =============================================================================

_OUTPUT_FORMATS: dict[str, str] = {
    "analyze": ANALYSIS_OUTPUT_FORMAT,
    "analysis": ANALYSIS_OUTPUT_FORMAT,
    "plan": PLAN_OUTPUT_FORMAT,
    "planning": PLAN_OUTPUT_FORMAT,
    "implement": IMPLEMENTATION_OUTPUT_FORMAT,
    "implementation": IMPLEMENTATION_OUTPUT_FORMAT,
    "review": REVIEW_OUTPUT_FORMAT,
    "research": RESEARCH_OUTPUT_FORMAT,
    "test": TEST_OUTPUT_FORMAT,
    "testing": TEST_OUTPUT_FORMAT,
    "default": DEFAULT_OUTPUT_FORMAT,
}

# Role-specific format overrides
_ROLE_FORMATS: dict[tuple[Role, str], str] = {
    (Role.EXPLORER, "default"): ANALYSIS_OUTPUT_FORMAT,
    (Role.PLANNER, "default"): PLAN_OUTPUT_FORMAT,
    (Role.CODER, "default"): IMPLEMENTATION_OUTPUT_FORMAT,
    (Role.CRITIC, "default"): REVIEW_OUTPUT_FORMAT,
    (Role.SME, "default"): RESEARCH_OUTPUT_FORMAT,
    (Role.TEST_ENGINEER, "default"): TEST_OUTPUT_FORMAT,
    (Role.REVIEWER, "default"): REVIEW_OUTPUT_FORMAT,
}


def get_output_format(capability: str, role: Optional[Role] = None) -> str:
    """Get the output format specification for a capability/role.
    
    Args:
        capability: The capability being executed
        role: Optional role for role-specific format
        
    Returns:
        Output format specification string
    """
    # Check role-specific format first
    if role:
        role_format = _ROLE_FORMATS.get((role, capability))
        if role_format:
            return role_format
        
        role_format = _ROLE_FORMATS.get((role, "default"))
        if role_format:
            return role_format
    
    # Fall back to capability-specific format
    return _OUTPUT_FORMATS.get(capability, _OUTPUT_FORMATS.get(capability.lower(), DEFAULT_OUTPUT_FORMAT))
