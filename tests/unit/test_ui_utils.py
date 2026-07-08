import pytest
from spine.ui.utils import format_bytes


def test_format_bytes_zero():
    assert format_bytes(0) == "0 B"


def test_format_bytes_sub_1024():
    for i in range(1, 1024):
        result = format_bytes(i)
        assert result.endswith(" B")
        assert result.split()[0] == str(i)


def test_format_bytes_kb_range():
    result = format_bytes(1024)
    assert result == "1.0 KB"


def test_format_bytes_mb_range():
    result = format_bytes(1024 * 1024)
    assert result == "1.0 MB"


def test_format_bytes_gb_range():
    result = format_bytes(1024 * 1024 * 1024)
    assert result == "1.0 GB"
