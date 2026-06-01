"""Tests for ``suppress_parsed_serializer_warning`` (the benign ``parsed`` filter).

``with_structured_output`` round-trips through provider response models whose
``parsed`` field is a generic ``Optional[...]``; serialising one carrying a
structured instance makes pydantic-core emit a multi-line ``UserWarning`` that
floods the logs during onboarding synthesis. The onboarding nodes wrap their
``ainvoke`` calls in this suppressor — but the ORIGINAL inline filters never
matched the warning (``warnings.filterwarnings`` matches with an anchored
``re.match`` and ``.`` does not cross newlines, so ``.*PydanticSerialization...``
could not get past the first ``\\n``). These tests pin the regression and verify
the fix is both effective and targeted.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.agents.helpers import suppress_parsed_serializer_warning

# The exact multi-line text pydantic-core emits for the ``parsed`` field.
_REAL_WARNING = (
    "Pydantic serializer warnings:\n"
    "  PydanticSerializationUnexpectedValue(Expected `none` - serialized value "
    "may not be as expected [field_name='parsed', "
    "input_value=SectionResult(doc_id='ARCHITECTURE_MAP', status='success'), "
    "input_type=SectionResult])"
)


def test_suppresses_the_real_parsed_warning() -> None:
    """The real multi-line ``parsed`` warning is swallowed inside the context."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with suppress_parsed_serializer_warning():
            warnings.warn(_REAL_WARNING, UserWarning)
    assert caught == [], "the parsed serializer warning should be suppressed"


def test_does_not_suppress_unrelated_warnings() -> None:
    """The filter is targeted — other warnings (incl. non-``parsed`` serializer
    warnings) still surface, so a genuine problem is never hidden."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with suppress_parsed_serializer_warning():
            warnings.warn("some unrelated UserWarning", UserWarning)
            warnings.warn(
                "Pydantic serializer warnings:\n  Unexpected field_name='foo'",
                UserWarning,
            )
    messages = [str(w.message) for w in caught]
    assert any("some unrelated UserWarning" in m for m in messages)
    # A serializer warning that is NOT about the ``parsed`` field is left alone.
    assert any("field_name='foo'" in m for m in messages)


def test_filter_restored_on_exit() -> None:
    """The suppression is scoped — the warning fires again after the block."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with suppress_parsed_serializer_warning():
            pass
        warnings.warn(_REAL_WARNING, UserWarning)
    assert len(caught) == 1


def test_regression_old_regex_could_not_match_multiline_message() -> None:
    """Document WHY the original inline filters failed: anchored ``re.match`` +
    no DOTALL means ``.*`` cannot cross the newline to reach the field name."""
    old_pattern = r".*PydanticSerializationUnexpectedValue.*parsed.*"
    assert re.compile(old_pattern, re.I).match(_REAL_WARNING) is None

    fixed_pattern = r"(?s)Pydantic serializer warnings.*parsed"
    assert re.compile(fixed_pattern, re.I).match(_REAL_WARNING) is not None
