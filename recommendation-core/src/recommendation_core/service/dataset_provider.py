from datetime import datetime
from pathlib import Path

import logging
import pandas as pd
from feast import FeatureStore
from recommendation_core.models import data_util

logger = logging.getLogger(__name__)


class DatasetProvider:

    def __init__(self, data_dir, force_load):
        self._item_df_path = Path(data_dir) / "recommendation_items.parquet"
        self._user_df_path = Path(data_dir) / "recommendation_users.parquet"
        self._interaction_df_path = (
                Path(data_dir) / "recommendation_interactions.parquet"
        )
        self._loaded = False

        if (
                self._item_df_path.exists() and self._user_df_path.exists()
                and self._interaction_df_path.exists() and (force_load is False)
        ):
            self._item_df = pd.read_parquet(self._item_df_path)
            self._user_df = pd.read_parquet(self._user_df_path)
            self._interaction_df = pd.read_parquet(self._interaction_df_path)
            self._loaded = True

    def item_df(self):
        return self._item_df

    def user_df(self):
        return self._user_df

    def interaction_df(self):
        return self._interaction_df

    def _save_dfs_to_parquet(self):
        self._item_df.to_parquet(self._item_df_path)
        self._user_df.to_parquet(self._user_df_path)
        self._interaction_df.to_parquet(self._interaction_df_path)


class LocalDatasetProvider(DatasetProvider):

    def __init__(self, store=None, data_dir="./src/recommendation_core/feature_repo/data"):
        super().__init__(data_dir, False)
        if self._loaded is False:
            assert store is not None
            self._load_from_store(store)
            self._save_dfs_to_parquet()

    def _load_from_store(self, store: FeatureStore):
        # load feature services
        item_service = store.get_feature_service("item_service")
        user_service = store.get_feature_service("user_service")
        interaction_service = store.get_feature_service("interaction_service")
        logger.info("service loaded")

        interactions_ids = pd.read_parquet(
            self._interaction_df_path
        )  # this line will create bug in new dataset case, wait for Feast bug fix
        user_ids = interactions_ids["user_id"].unique().tolist()
        item_ids = interactions_ids["item_id"].unique().tolist()
        # select which items to use for the training
        item_entity_df = pd.DataFrame.from_dict(
            {
                "item_id": item_ids,
                "event_timestamp": [datetime(2025, 1, 1)] * len(item_ids),
            }
        )
        # select which users to use for the training
        user_entity_df = pd.DataFrame.from_dict(
            {
                "user_id": user_ids,
                "event_timestamp": [datetime(2025, 1, 1)] * len(user_ids),
            }
        )
        # Select which item-user interactions to use for the training
        item_user_interactions_df = interactions_ids[["item_id", "user_id"]].copy()
        item_user_interactions_df["event_timestamp"] = datetime(2025, 1, 1)
        # retrieve datasets for training
        self._item_df = store.get_historical_features(
            entity_df=item_entity_df, features=item_service
        ).to_df()
        self._user_df = store.get_historical_features(
            entity_df=user_entity_df, features=user_service
        ).to_df()
        self._interaction_df = store.get_historical_features(
            entity_df=item_user_interactions_df, features=interaction_service
        ).to_df()


class RemoteDatasetProvider(DatasetProvider):

    def __init__(self, url: str, data_dir="./src/recommendation_core/feature_repo/data", force_load=False):
        super().__init__(data_dir, force_load)
        if self._loaded is False:
            df = pd.read_csv(url)
            self._item_df, self._user_df, self._interaction_df = data_util.clean_dataset(df)
            self._save_dfs_to_parquet()


