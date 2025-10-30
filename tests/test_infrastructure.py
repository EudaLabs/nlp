"""Basic tests to verify test infrastructure is working properly."""
import sys

import pytest


def test_python_version():
    """Verify Python version is 3.8 or higher."""
    assert sys.version_info >= (3, 8), f"Python version is {sys.version_info}"


def test_imports_basic():
    """Test that we can import common libraries."""
    try:
        import numpy
        import pandas
        
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import basic libraries: {e}")


def test_imports_nlp():
    """Test that we can import NLP libraries."""
    try:
        import sklearn
        
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import NLP libraries: {e}")
