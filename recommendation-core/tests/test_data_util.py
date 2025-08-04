from datetime import datetime

import pandas as pd
import pytest
import torch

from models.data_util import data_preproccess


@pytest.fixture
def sample_dataframe():
    # Create a sample DataFrame with different types of features
    df = pd.DataFrame(
        {
            # Numerical features
            "price": [100.0, 200.0, 300.0],
            "quantity": [1, 2, 3],
            # Datetime features
            "created_at": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            # Categorical features (low cardinality)
            "category": ["A", "B", "A"],
            "status": ["active", "inactive", "active"],
            # Text features
            "description": [
                "This is a product description",
                "Another product description",
                "Third product description",
            ],
            # URL features
            "image_url": [
                "http://example.com/image1.jpg",
                "http://example.com/image2.jpg",
                "http://example.com/image3.jpg",
            ],
            # Features to be filtered out
            "user_id": ["u1", "u2", "u3"],
            "event_timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
        }
    )
    return df


def test_data_preproccess_basic(sample_dataframe):
    # Process the data
    result = data_preproccess(sample_dataframe)

    # Check that the result is a dictionary with expected keys
    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "numerical_features",
        "categorical_features",
        "text_features",
        "url_image",
    }

    # Check numerical features
    assert isinstance(result["numerical_features"], torch.Tensor)
    assert result["numerical_features"].shape[0] == len(sample_dataframe)
    assert result["numerical_features"].shape[1] == 3  # price, quantity, created_at

    # Check categorical features
    assert isinstance(result["categorical_features"], torch.Tensor)
    assert result["categorical_features"].shape[0] == len(sample_dataframe)
    assert result["categorical_features"].shape[1] == 2  # category, status

    # Check text features
    assert isinstance(result["text_features"], torch.Tensor)
    assert result["text_features"].shape[0] == len(sample_dataframe)
    assert result["text_features"].shape[1] == 1  # description
    assert result["text_features"].shape[2] == 384  # BGE model embedding dimension

    # Check URL features
    assert isinstance(result["url_image"], list)
    assert len(result["url_image"]) == len(sample_dataframe)
    assert len(result["url_image"][0]) == 1  # image_url
