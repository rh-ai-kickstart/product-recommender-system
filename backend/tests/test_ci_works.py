"""
Simple tests that will always pass to ensure CI pipeline works.
"""

import sys


def test_python_version():
    """Test that we're running a compatible Python version."""
    assert sys.version_info >= (3, 8), f"Python version {sys.version} is too old"


def test_basic_math():
    """Test basic Python functionality."""
    assert 2 + 2 == 4
    assert "hello" + " world" == "hello world"


def test_import_basic_modules():
    """Test that basic Python modules can be imported."""
    import os
    import json
    import pathlib
    
    assert os is not None
    assert json is not None
    assert pathlib is not None


def test_pytest_working():
    """Test that pytest itself is working."""
    assert True, "Pytest is working correctly"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__]) 