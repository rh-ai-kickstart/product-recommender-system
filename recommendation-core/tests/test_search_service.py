from unittest.mock import Mock, patch

import pandas as pd
import pytest
import torch

from service.search_by_text import SearchService


@pytest.fixture
def mock_feature_store():
    store = Mock()
    # Mock the retrieve_online_documents method to return a DataFrame
    store.retrieve_online_documents.return_value.to_df.return_value = pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "event_timestamp": ["2024-01-01", "2024-01-01", "2024-01-01"],
        }
    )
    # Mock the get_feature_service method
    store.get_feature_service.return_value = Mock()
    # Mock the get_historical_features method
    store.get_historical_features.return_value.to_df.return_value = pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "title": ["Item 1", "Item 2", "Item 3"],
            "description": ["Desc 1", "Desc 2", "Desc 3"],
        }
    )
    return store


@pytest.fixture
def search_service(mock_feature_store):
    with (
        patch("service.search_by_text.AutoTokenizer") as mock_tokenizer,
        patch("service.search_by_text.AutoModel") as mock_model,
    ):
        # Mock the tokenizer
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Mock the model
        mock_model.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value.return_value = (
            torch.tensor([[[0.1, 0.2, 0.3]]]),  # last_hidden_state
            None,  # pooler_output
        )

        service = SearchService(mock_feature_store)
        return service


def test_search_by_text(search_service, mock_feature_store):
    # Test the search functionality
    results = search_service.search_by_text("test query", k=3)

    # Verify the results
    assert len(results) == 3
    assert "item_id" in results.columns
    assert "title" in results.columns
    assert "description" in results.columns

    # Verify the mock calls
    mock_feature_store.retrieve_online_documents.assert_called_once()
    mock_feature_store.get_feature_service.assert_called_once_with("item_service")
    mock_feature_store.get_historical_features.assert_called_once()


def test_search_by_text_empty_results(search_service, mock_feature_store):
    # Mock empty results
    mock_feature_store.retrieve_online_documents.return_value.to_df.return_value = (
        pd.DataFrame(columns=["item_id", "event_timestamp"])
    )
    mock_feature_store.get_historical_features.return_value.to_df.return_value = (
        pd.DataFrame(columns=["item_id", "title", "description"])
    )

    # Test the search functionality with empty results
    results = search_service.search_by_text("test query", k=3)

    # Verify the results
    assert len(results) == 0
    assert "item_id" in results.columns
    assert "title" in results.columns
    assert "description" in results.columns
