import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from kafka import KafkaProducer


class KafkaService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KafkaService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the Kafka producer"""
        kafka_service = os.getenv(
            "KAFKA_SERVICE_ADDR",
            "rec-sys-cluster-kafka-bootstrap.rec-sys.svc.cluster.local:9092",
        )
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_service,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def _build_interaction_schema(self) -> Dict[str, Any]:
        """Build the schema for interaction messages"""
        return {
            "type": "struct",
            "fields": [
                {
                    "field": "user_id",
                    "type": "string",
                    "optional": False,
                },
                {
                    "field": "item_id",
                    "type": "string",
                    "optional": False,
                },
                {
                    "field": "timestamp",
                    "type": "string",
                    "optional": False,
                    "format": "timestamp",
                },
                {
                    "field": "interaction_type",
                    "type": "string",
                    "optional": False,
                },
                {
                    "field": "rating",
                    "type": "float64",
                    "optional": True,
                },
                {
                    "field": "quantity",
                    "type": "float64",
                    "optional": True,
                },
                {
                    "field": "interaction_id",
                    "type": "string",
                    "optional": False,
                },
                {
                    "field": "review_title",
                    "type": "string",
                    "optional": True,
                },
                {
                    "field": "review_content",
                    "type": "string",
                    "optional": True,
                },
            ],
            "optional": False,
            "name": "interaction",
        }

    def _build_new_user_schema(self) -> Dict[str, Any]:
        """Build the schema for new user messages"""
        return {
            "type": "struct",
            "fields": [
                {
                    "field": "user_id",
                    "type": "string",
                    "optional": False,
                },
                {
                    "field": "user_name",
                    "type": "string",
                    "optional": False,
                },
                {
                    "field": "preferences",
                    "type": "string",
                    "optional": False,
                },
                {
                    "field": "signup_date",
                    "type": "string",
                    "optional": False,
                    "format": "timestamp",
                },
            ],
            "optional": False,
            "name": "new-users",
        }

    def send_interaction(
        self,
        user_id: str,
        item_id: str,
        interaction_type: str,
        rating: Optional[int] = None,
        quantity: Optional[int] = None,
        review_title: Optional[str] = None,
        review_content: Optional[str] = None,
    ) -> None:
        """Send an interaction event to Kafka"""
        schema = self._build_interaction_schema()
        interaction = {
            "user_id": str(user_id),
            "item_id": item_id,
            "timestamp": datetime.now().isoformat(" "),
            "interaction_type": interaction_type,
            "rating": int(rating) if rating is not None else None,
            "quantity": int(quantity) if quantity is not None else None,
            "review_title": review_title if review_title is not None else "",
            "review_content": review_content if review_content is not None else "",
            "interaction_id": f"{user_id}-{item_id}-{datetime.now(timezone.utc).timestamp()}",
            # example unique ID
        }
        message = {"schema": schema, "payload": interaction}
        self.producer.send("interactions", message)
        self.producer.flush()

    def send_new_user(self, user_id: Union[int, str], user_name: str, preferences: str) -> None:
        """Send a new user event to Kafka"""
        schema = self._build_new_user_schema()
        user_data = {
            "user_id": str(user_id),
            "user_name": str(user_name),
            "preferences": str(preferences),
            "signup_date": datetime.now().isoformat(" "),
        }
        message = {"schema": schema, "payload": user_data}
        self.producer.send("new-users", message)
        self.producer.flush()
