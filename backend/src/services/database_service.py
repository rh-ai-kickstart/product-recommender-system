import logging
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from database.models_sql import StreamInteraction

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service to handle direct database writes (replaces Kafka)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._initialized = True

    async def log_interaction(
        self,
        db: AsyncSession,
        user_id: str,
        item_id: str,
        interaction_type: str,
        rating: Optional[int] = None,
        quantity: Optional[int] = None,
        review_title: Optional[str] = None,
        review_content: Optional[str] = None,
    ) -> None:
        """Log an interaction directly to the database"""
        try:
            interaction_id = f"{user_id}-{item_id}-{datetime.now(timezone.utc).timestamp()}"

            interaction = StreamInteraction(
                user_id=str(user_id),
                item_id=item_id,
                timestamp=datetime.now(),
                interaction_type=interaction_type,
                rating=float(rating) if rating is not None else None,
                quantity=float(quantity) if quantity is not None else None,
                review_title=review_title if review_title is not None else "",
                review_content=review_content if review_content is not None else "",
                interaction_id=interaction_id,
            )

            db.add(interaction)
            await db.commit()

            logger.info(f"Interaction logged to database: {interaction_id}")

        except Exception as e:
            logger.error(f"Failed to log interaction to database: {e}")
            await db.rollback()
            raise


# Global instance - can be imported and used everywhere
db_service = DatabaseService()
