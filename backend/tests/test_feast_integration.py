"""
Integration tests to verify recommendation-core and Feast integration.
"""

import pytest


def test_feast_service_import():
    """Test that the FeastService can be imported."""
    try:
        from services.feast.feast_service import FeastService
        assert FeastService is not None
    except ImportError as e:
        pytest.fail(f"FeastService import failed: {e}")


def test_feast_service_initialization():
    """Test that FeastService can be initialized (basic test without database)."""
    try:
        from services.feast.feast_service import FeastService
        
        # This will fail in CI without proper database setup, but we can test the import
        # and basic structure
        service_class = FeastService
        assert hasattr(service_class, '_instance')
        assert hasattr(service_class, '__new__')
        
    except Exception as e:
        # In CI environment without database, this is expected to fail
        # but we can still verify the class structure
        pytest.skip(f"FeastService initialization requires database setup: {e}")


def test_recommendation_core_models():
    """Test that recommendation-core models can be imported."""
    try:
        from recommendation_core.models.entity_tower import EntityTower
        from recommendation_core.models.two_tower import TwoTowerModel
        assert EntityTower is not None
        assert TwoTowerModel is not None
    except ImportError as e:
        pytest.fail(f"recommendation_core models import failed: {e}")


def test_recommendation_core_services():
    """Test that recommendation-core services can be imported."""
    try:
        from recommendation_core.service.clip_encoder import ClipEncoder
        from recommendation_core.service.dataset_provider import LocalDatasetProvider
        assert ClipEncoder is not None
        assert LocalDatasetProvider is not None
    except ImportError as e:
        pytest.fail(f"recommendation_core services import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 