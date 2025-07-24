import json
import os
from io import BytesIO
from pathlib import Path
from typing import List

import pandas as pd
import requests
import torch
from feast import FeatureStore
from minio import Minio
from models import Product, User
from PIL import Image as PILImage
from recsysapp.models.data_util import data_preproccess
from recsysapp.models.entity_tower import EntityTower
from recsysapp.service.clip_encoder import ClipEncoder
from recsysapp.service.dataset_provider import LocalDatasetProvider
from recsysapp.service.search_by_image import SearchByImageService
from recsysapp.service.search_by_text import SearchService

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_MODEL_SIZE = 512


class FeastService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FeastService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:

            self.store = FeatureStore(str(Path(__file__).parent))
            self._initialized = True
            self.user_encoder = self._load_user_encoder()
            self.user_service = self.store.get_feature_service("user_service")
            self.dataset_provider = LocalDatasetProvider(
                self.store, data_dir="services/feast/data"
            )  # TODO: remove path when Feast is the issue
            print("[Feast] Initializing ClipEncoder...")
            self.clip_encoder = ClipEncoder()
            print("[Feast] ClipEncoder initialized.")
            print("[Feast] Initializing SearchByImageService...")
            self.search_by_image_service = SearchByImageService(self.store, self.clip_encoder)
            print("[Feast] SearchByImageService initialized.")

    def _load_model_version(self):
        """
        Retrieve the most recently updated model version from the database.
        """
        from sqlalchemy import create_engine, text

        database_url = os.getenv("DATABASE_URL")
        engine = create_engine(database_url)

        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT version FROM model_version ORDER BY updated_at DESC LIMIT 1")
            )
            version = result.fetchone()[0]
            return version

    def _load_user_encoder(self):
        """
        Download and load the user encoder model and its configuration from MinIO.
        """
        minio_client = Minio(
            endpoint=os.getenv("MINIO_HOST", "endpoint") + ":" + os.getenv("MINIO_PORT", "9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "access-key"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "secret-key"),
            secure=False,
        )
        model_version = self._load_model_version()
        bucket_name = "user-encoder"
        object_name = f"user-encoder-{model_version}.pth"
        configuration = f"user-encoder-config-{model_version}.json"

        minio_client.fget_object(bucket_name, object_name, "/tmp/user-encoder.pth")
        minio_client.fget_object(bucket_name, configuration, "/tmp/user-encoder-config.json")

        with open("/tmp/user-encoder-config.json", "r") as f:
            json_config = json.load(f)
        user_encoder = EntityTower(
            json_config["users_num_numerical"], json_config["users_num_categorical"]
        )
        user_encoder.load_state_dict(torch.load("/tmp/user-encoder.pth"))

        return user_encoder

    def get_all_existing_users(self) -> List[dict]:
        """
        Return all existing user feature rows from the dataset provider.
        """
        try:
            user_df = self.dataset_provider.user_df()
            print("Fetched all users")
            return user_df
        except Exception as e:
            print(f"Failed to fetch users from feature view: {e}")
            return []

    def load_items_existing_user(self, user_id: str) -> List[Product]:
        """
        Get top recommended items for an existing user based on their user_id.
        """
        suggested_item_ids = self.store.get_online_features(
            features=self.store.get_feature_service("user_top_k_items"),
            entity_rows=[{"user_id": user_id}],
        )
        top_item_ids = suggested_item_ids.to_df().iloc[0]["top_k_item_ids"]
        return self._item_ids_to_product_list(top_item_ids)

    def load_items_new_user(self, user: User, k: int = 10):
        """
        Generate recommendations for a new user by encoding their features
        and querying the feature store for top-k similar items.
        """
        user_as_df = pd.DataFrame([user.model_dump()])
        user_embed = self.user_encoder(**data_preproccess(user_as_df))[0]
        top_k = self.store.retrieve_online_documents(
            query=user_embed.tolist(), top_k=k, features=["item_embedding:item_id"]
        )
        print("Retrieved documents from store:", top_k.to_df())
        top_item_ids = top_k.to_df()["item_id"].tolist()
        return self._item_ids_to_product_list(top_item_ids)

    def _item_ids_to_product_list(self, top_item_ids: pd.Series | List) -> List[Product]:
        """
        Given a list of item_ids, fetch and return full product details from the feature store.
        """
        suggested_item = self.store.get_online_features(
            features=self.store.get_feature_service("item_service"),
            entity_rows=[{"item_id": item_id} for item_id in top_item_ids],
        ).to_df()
        print(suggested_item.columns)
        print(suggested_item)
        suggested_item = [
            Product(
                item_id=row.item_id,
                product_name=row.product_name,
                category=row.category,
                about_product=getattr(row, "about_product", None),
                img_link=getattr(row, "img_link", None),
                discount_percentage=getattr(row, "discount_percentage", None),
                discounted_price=getattr(row, "discounted_price", None),
                actual_price=row.actual_price,
                product_link=getattr(row, "product_link", None),
                rating_count=getattr(row, "rating_count", None),
                rating=getattr(row, "rating", None),
            )
            for row in suggested_item.itertuples()
        ]
        return suggested_item

    def search_item_by_text(self, text: str, k=5):
        """
        Perform a semantic search over item descriptions using a text query.
        Returns top-k matching items.
        """
        search_service = SearchService(self.store)
        results_df = search_service.search_by_text(text, k)
        print(results_df)
        top_item_ids = results_df["item_id"].tolist()
        results = self._item_ids_to_product_list(top_item_ids)
        return results

    def search_item_by_image_link(self, image_link: str, k=5):
        """
        Perform image-based product search using an image URL.
        Returns top-k similar items.
        """
        try:
            # Manually check if the image is reachable and decodable
            resp = requests.get(image_link, timeout=100)
            resp.raise_for_status()

            # Try decoding it to ensure it's a valid image
            img = PILImage.open(BytesIO(resp.content))
            img.verify()  # Raises if the image is corrupt

        except Exception as e:
            print(f"[Validation] Could not fetch/validate image: {e}")
            raise ValueError("Invalid or unreachable image URL.")
        try:
            results_df = self.search_by_image_service.search_by_image_link(image_link, k)
            print(results_df)
            top_item_ids = results_df["item_id"].tolist()
            return self._item_ids_to_product_list(top_item_ids)
        except Exception as e:
            print(f"Error in search_item_by_image_link: {e}")
            raise ValueError("Failed to process image from URL.")

    def search_item_by_image_file(self, image: PILImage.Image, k=5):
        print("[Feast] Starting search_item_by_image_file")
        try:
            results_df = self.search_by_image_service.search_by_image(image, k)
            print("[Feast] search_by_image() completed")
            print(results_df)

            if results_df.empty or "item_id" not in results_df:
                raise ValueError("No valid item_id results returned from image search.")

            top_item_ids = results_df["item_id"].tolist()
            return self._item_ids_to_product_list(top_item_ids)

        except Exception as e:
            print(f"[SearchByImage Error] {e}")
            raise ValueError("Failed to process image.")

    def get_item_by_id(self, item_id: int) -> Product:
        """
        Retrieve a single item by its ID and return it as a Product
        """
        product_list = self._item_ids_to_product_list([item_id])
        if not product_list:
            raise ValueError(f"Item with ID {item_id} not found.")
        return product_list[0]
