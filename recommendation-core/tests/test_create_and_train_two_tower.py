import pandas as pd
import pytest

from models.entity_tower import EntityTower
from models.train_two_tower import create_and_train_two_tower


@pytest.fixture
def sample_dataframes():
    # Create sample DataFrames for items, users, and interactions
    item_df = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3"],
            "price": [100.0, 200.0, 300.0],
            "category": ["A", "B", "A"],
            "description": ["Item 1", "Item 2", "Item 3"],
            "image_url": [
                "http://example.com/1.jpg",
                "http://example.com/2.jpg",
                "http://example.com/3.jpg",
            ],
        }
    )

    user_df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "age": [25, 30, 35],
            "preferences": ["X", "Y", "X"],
            "bio": ["User 1", "User 2", "User 3"],
        }
    )

    interaction_df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "item_id": ["i1", "i2", "i3"],
            "rating": [4.0, 3.0, 5.0],
            "quantity": [1, 2, 1],
            "interaction_type": ["purchase", "view", "purchase"],
        }
    )

    return item_df, user_df, interaction_df


def test_create_and_train_two_tower_basic(sample_dataframes):
    item_df, user_df, interaction_df = sample_dataframes

    # Test basic training
    item_tower, user_tower = create_and_train_two_tower(
        item_df,
        user_df,
        interaction_df,
        n_epochs=2,  # Use small number of epochs for testing
    )

    # Check that towers are created and trained
    assert isinstance(item_tower, EntityTower)
    assert isinstance(user_tower, EntityTower)

    # Check that towers have the correct number of features
    assert item_tower.num_numerical > 0
    assert item_tower.num_of_categories > 0
    assert user_tower.num_numerical > 0
    assert user_tower.num_of_categories > 0


def test_create_and_train_two_tower_with_losses(sample_dataframes):
    item_df, user_df, interaction_df = sample_dataframes

    # Test training with loss tracking
    item_tower, user_tower, epoch_losses = create_and_train_two_tower(
        item_df, user_df, interaction_df, return_epoch_losses=True, n_epochs=2
    )

    # Check losses
    assert isinstance(epoch_losses, list)
    assert len(epoch_losses) == 2
    assert all(isinstance(loss, float) for loss in epoch_losses)
    assert all(loss >= 0 for loss in epoch_losses)


def test_create_and_train_two_tower_with_model_definition(sample_dataframes):
    item_df, user_df, interaction_df = sample_dataframes

    # Test training with model definition
    item_tower, user_tower, model_def = create_and_train_two_tower(
        item_df, user_df, interaction_df, return_model_definition=True, n_epochs=2
    )

    # Check model definition
    assert isinstance(model_def, dict)
    assert "items_num_numerical" in model_def
    assert "items_num_categorical" in model_def
    assert "users_num_numerical" in model_def
    assert "users_num_categorical" in model_def
    assert all(isinstance(v, int) for v in model_def.values())


def test_create_and_train_two_tower_with_all_returns(sample_dataframes):
    item_df, user_df, interaction_df = sample_dataframes

    # Test training with all return values
    item_tower, user_tower, epoch_losses, model_def = create_and_train_two_tower(
        item_df,
        user_df,
        interaction_df,
        return_epoch_losses=True,
        return_model_definition=True,
        n_epochs=2,
    )

    # Check all return values
    assert isinstance(item_tower, EntityTower)
    assert isinstance(user_tower, EntityTower)
    assert isinstance(epoch_losses, list)
    assert isinstance(model_def, dict)
    assert len(epoch_losses) == 2
    assert all(isinstance(loss, float) for loss in epoch_losses)
    assert all(isinstance(v, int) for v in model_def.values())
