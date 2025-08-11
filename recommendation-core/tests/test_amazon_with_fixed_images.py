from pathlib import Path

import pandas as pd
import pytest

from generation.amazon_with_fixed_images import fix_url


@pytest.fixture
def wrong_url():
    return 'https://m.media-amazon.com/images/W/WEBP_402378-T1/images/I/51UsScvHQNL._SX300_SY300_QL70_FMwebp_.jpg'


@pytest.fixture
def correct_url():
    return 'https://m.media-amazon.com/images/I/51UsScvHQNL._SX300_SY300_QL70_FMwebp_.jpg'


def test_fix_url(wrong_url, correct_url):
    assert fix_url(correct_url) == (correct_url, False)
    assert fix_url(wrong_url) == (correct_url, True)


@pytest.fixture()
def base_path():
    return Path(__file__).parent.parent.joinpath('src').joinpath('feature_repo').joinpath('data')


@pytest.fixture
def amazon_df(base_path):
    return pd.read_csv(base_path.joinpath("amazon.csv"))


@pytest.fixture
def amazon_with_fixed_images_df(base_path):
    return pd.read_csv(base_path.joinpath("amazon_with_fixed_images.csv"))


def test_image_replacement(amazon_df, amazon_with_fixed_images_df):
    assert amazon_with_fixed_images_df.shape == amazon_df.shape
