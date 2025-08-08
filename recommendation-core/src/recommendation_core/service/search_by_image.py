import pandas as pd
import requests
from feast import FeatureStore
from PIL import Image

from recommendation_core.service.clip_encoder import ClipEncoder


class SearchByImageService:
    def __init__(self, store: FeatureStore, clip_encoder: ClipEncoder):
        self.store = store
        self.clip_encoder = clip_encoder

    def search_by_image_link(self, image_link, k):
        image = Image.open(requests.get(image_link, stream=True).raw)
        return self.search_by_image(image, k)

    def search_by_image(self, image, k):
        clip_embedding = (
            self.clip_encoder.encode_images([image])[0].cpu().detach().numpy()
        )

        ids = self.store.retrieve_online_documents(
            query=list(clip_embedding),
            top_k=k,
            features=["item_clip_features_embed:item_id"],
        ).to_df()
        ids["event_timestamp"] = pd.to_datetime("now", utc=True)

        item_service = self.store.get_feature_service("item_service")
        values = self.store.get_historical_features(
            entity_df=ids,
            features=item_service,
        ).to_df()
        return values
