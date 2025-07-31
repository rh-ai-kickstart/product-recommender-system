from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from database.models_sql import User
from routes.auth import get_current_user

router = APIRouter(prefix="/interactions", tags=["interactions"])


class InteractionRequest(BaseModel):
    item_id: str
    interaction_type: str
    rating: int
    review_title: str
    review_content: str
    quantity: int


@router.post("", status_code=status.HTTP_201_CREATED)
async def log_interaction(
    interaction: InteractionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    interaction_id = str(uuid4())
    timestamp = datetime.now(timezone.utc)

    # Simulated log (to be replaced with Kafka in the future)
    print(
        {
            "interaction_id": interaction_id,
            "user_id": user.user_id,
            "item_id": interaction.item_id,
            "interaction_type": interaction.interaction_type,
            "rating": interaction.rating,
            "review_title": interaction.review_title,
            "review_content": interaction.review_content,
            "quantity": interaction.quantity,
            "timestamp": timestamp.isoformat(),
        }
    )

    return {
        "message": "Interaction logged (simulated)",
        "interaction_id": interaction_id,
    }
