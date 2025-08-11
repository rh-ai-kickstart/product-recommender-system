import pytest
import torch

from models.entity_tower import EntityTower

D_MODEL = 64
DIM_RATIO = {"numeric": 1, "categorical": 2, "text": 7, "image": 0}


@pytest.fixture
def sample_entity_tower():
    return EntityTower(
        num_numerical=3,
        num_of_categories=10,
        d_model=D_MODEL,
        text_embed_dim=384,
        image_embed_dim=384,
        dim_ratio=DIM_RATIO,
    )


@pytest.fixture
def sample_batch():
    batch_size = 4
    return {
        "numerical_features": torch.randn(batch_size, 3),  # 3 numerical features
        "categorical_features": torch.randint(
            0, 10, (batch_size, 1)
        ),  # always use a single categorical_features
        "text_features": torch.randn(
            batch_size, 1, 384
        ),  # 1 text feature with 384-dim embedding
        "url_image": [
            ["http://example.com/image1.jpg"] for _ in range(batch_size)
        ],  # 1 URL per sample
    }


def test_entity_tower_initialization(sample_entity_tower):
    # Test initialization parameters
    assert sample_entity_tower.num_numerical == 3
    assert sample_entity_tower.num_of_categories == 10
    # Calculate expected text dimension based on EntityTower logic

    num_numerical = sample_entity_tower.num_numerical
    num_of_categories = sample_entity_tower.num_of_categories
    ratio_weight = D_MODEL / sum(DIM_RATIO.values())
    numerical_dim = int(DIM_RATIO["numeric"] * ratio_weight) if num_numerical > 0 else 0
    categorical_dim = (
        int(DIM_RATIO["categorical"] * ratio_weight) if num_of_categories > 0 else 0
    )
    expected_text_dim = D_MODEL - numerical_dim - categorical_dim
    assert sample_entity_tower.text_dim == expected_text_dim

    # Test that required modules are created
    assert isinstance(sample_entity_tower.categorical_embed, torch.nn.Embedding)
    assert isinstance(sample_entity_tower.numeric_norm, torch.nn.BatchNorm1d)
    assert isinstance(sample_entity_tower.numeric_embed, torch.nn.Linear)
    assert isinstance(sample_entity_tower.project_text, torch.nn.Linear)
    assert isinstance(sample_entity_tower.fn1, torch.nn.Linear)
    assert isinstance(sample_entity_tower.fn2, torch.nn.Linear)
    assert isinstance(sample_entity_tower.norm, torch.nn.RMSNorm)
    assert isinstance(sample_entity_tower.norm1, torch.nn.RMSNorm)


def test_entity_tower_forward(sample_entity_tower, sample_batch):
    output = sample_entity_tower(
        sample_batch["numerical_features"],
        sample_batch["categorical_features"],
        sample_batch["text_features"],
        sample_batch["url_image"],
    )

    # Check output shape
    assert output.shape == (4, 64)  # batch_size, d_model

    # Check that output is not NaN
    assert not torch.isnan(output).any()

    # Check that output is not zero
    assert not torch.allclose(output, torch.zeros_like(output))
