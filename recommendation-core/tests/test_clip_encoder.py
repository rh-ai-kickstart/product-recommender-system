import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
import torch
from PIL import Image

from service.clip_encoder import ClipEncoder


@pytest.fixture(scope="session", autouse=True)
def before_all(request):
    seed = 739
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@pytest.fixture
def item_df():
    parquet_file = (
        Path(__file__).parent.joinpath("data").joinpath("recommendation_items.parquet")
    )
    return pd.read_parquet(parquet_file)


@pytest.fixture
def clip_encoder():
    return ClipEncoder()


@pytest.fixture
def simple_texts():
    return ["a photo of a cat", "a photo of a dog"]


@pytest.fixture
def simple_images():
    image_links = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "https://farm1.staticflickr.com/111/299422173_a073c92714_z.jpg",
    ]
    images = [Image.open(requests.get(url, stream=True).raw) for url in image_links]

    generated_image_path = (
        Path(__file__)
        .parent.parent
        .joinpath("generation")
        .joinpath("data")
        .joinpath("generated_images")
    )
    gen_1 = generated_image_path.joinpath("item_CarVac Pro.png")
    gen_2 = generated_image_path.joinpath("item_FilterPro Set.png")

    images.append(Image.open(gen_1))
    images.append(Image.open(gen_2))
    return images


@pytest.fixture
def more_texts(simple_texts):
    return simple_texts * 10


@pytest.fixture
def more_images(simple_images):
    return simple_images * 5


@pytest.fixture
def images_having_nones(more_images: list):
    result = more_images.copy()
    result[7] = None
    result[3] = None
    result[9] = None
    return result


@pytest.fixture
def long_text():
    txt_path = Path(__file__).parent.joinpath("data").joinpath("long_text.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def test_text_encoding(clip_encoder, more_texts):
    result_batched = clip_encoder.encode_texts_batched(more_texts, batch_size=3)
    result_simple = clip_encoder.encode_texts(more_texts)
    assert torch.allclose(result_batched, result_simple, 1e-05, 1e-05)


def test_image_encoding(clip_encoder, more_images, images_having_nones):
    # non batched
    result_simple = clip_encoder.encode_images(more_images)
    # batched with no nones
    result_batched, none_indices = clip_encoder.encode_images_batched_having_nones(
        more_images, batch_size=3
    )
    assert none_indices == []
    # we expect the same result
    assert torch.allclose(result_batched, result_simple, 1e-05, 1e-05)

    # batched with nones
    embeddings, none_indices = clip_encoder.encode_images_batched_having_nones(
        images_having_nones, batch_size=3
    )
    assert none_indices == [3, 7, 9]
    # we expect the same results for non-nones
    for i, _ in enumerate(embeddings):
        if i not in none_indices:
            assert torch.allclose(embeddings[i], result_simple[i], 1e-05, 1e-05)


def test_image_and_text_encoding(clip_encoder, more_texts, images_having_nones):
    clip_embeddings = clip_encoder.encode_texts_and_images(
        more_texts, images_having_nones, 4
    )
    assert clip_embeddings is not None


def test_item_df_embedding_generation(clip_encoder, item_df):
    item_clip_features_embed = clip_encoder.clip_embeddings(item_df)
    # produced the object to be used by the workflow:
    # store.push('item_clip_features_embed', item_clip_features_embed, to=PushMode.ONLINE, allow_registry_cache=False)
    assert item_clip_features_embed is not None


@pytest.fixture
def item_wrong_img_df():
    return pd.DataFrame({
        'about_product': ['this is a nice product!'],
        'img_link': ['https://m.media-amazon.com/images/W/WEBP_402378-T1/images/I/51UsScvHQNL._SX300_SY300_QL70_FMwebp_.jpg']
    })


def test_wrong_url(clip_encoder, item_wrong_img_df):
    clip_embeddings = clip_encoder.create_clip_embeddings(item_wrong_img_df)
    assert clip_embeddings


def test_long_text(clip_encoder, long_text):
    embeddings = clip_encoder.encode_texts([long_text])
    assert embeddings is not None
