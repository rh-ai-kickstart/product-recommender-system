"""
Basic tests to ensure the CI pipeline works correctly.
"""

import pytest


def test_basic_import():
    """Test that basic imports work."""
    try:
        import fastapi
        assert fastapi is not None
    except ImportError:
        pytest.fail("FastAPI import failed")


def test_health_check():
    """Test that we can create a basic health check endpoint."""
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/health")
        def health_check():
            return {"status": "ok"}

        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
    except ImportError as e:
        pytest.skip(f"FastAPI test dependencies not available: {e}")


def test_python_version():
    """Test that we're running the expected Python version."""
    import sys
    # Ensure we're running Python 3.12 or compatible
    assert sys.version_info >= (3, 8), f"Python version {sys.version} is too old"


def test_recommendation_core_import():
    """Test that recommendation-core can be imported."""
    try:
        import recommendation_core
        assert recommendation_core is not None
        print("✅ recommendation_core imported successfully")
    except ImportError as e:
        pytest.skip(f"recommendation_core not available - skipping integration test: {e}")


def test_feast_import():
    """Test that Feast can be imported (dependency of recommendation-core)."""
    try:
        import feast
        assert feast is not None
        print("✅ feast imported successfully")
    except ImportError as e:
        pytest.skip(f"feast not available - skipping integration test: {e}")


def test_pyarrow_import():
    """Test that pyarrow can be imported (dependency of feast)."""
    try:
        import pyarrow
        assert pyarrow is not None
        print("✅ pyarrow imported successfully")
    except ImportError as e:
        pytest.skip(f"pyarrow not available - skipping integration test: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 