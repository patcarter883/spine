from spine.ui.utils import format_bytes
import pytest
from spine.ui.utils import truncate_middle


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



def test_truncate_middle():
    # Test when text length <= max_len
    assert truncate_middle('abcde', 5) == 'abcde'
    assert truncate_middle('hi', 10) == 'hi'
    
    # Test empty string
    assert truncate_middle('', 5) == ''
    assert truncate_middle('', 0) == ''
    
    # Test max_len < 3 (no ellipsis)
    assert truncate_middle('abcdef', 1) == 'a'
    assert truncate_middle('abcdef', 2) == 'ab'
    
    # Test max_len == 3
    result = truncate_middle('abcde', 3)
    assert len(result) == 3
    assert '…' in result
    assert result[0] == 'a' and result[-1] == 'e'
    
    # Test normal truncation with ellipsis (max_len >= 3)
    result = truncate_middle('abcdefghijk', 7)
    assert len(result) == 7
    assert '…' in result
    # Check that ellipsis is in the middle roughly
    assert result.count('…') == 1
    
    # Test very long string distribution
    result = truncate_middle('abcdefghijklmnopqrstuvwxyz', 10)
    assert len(result) == 10
    assert '…' in result
    # Verify characters from start and end are present
    assert result.startswith('a')
    assert result.endswith('z')


