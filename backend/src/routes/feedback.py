from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from models import Feedback
from models import InteractionType
from routes.auth import get_current_user
from services.database_service import db_service  # Use global instance

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback: Feedback,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit product feedback (rating and review)
    """
    try:
        # Log feedback interaction to database (replaces Kafka)
        await db_service.log_interaction(
            db=db,
            user_id=user.user_id,
            item_id=feedback.productId,
            interaction_type=InteractionType.RATE,
            rating=feedback.rating,
            review_title=feedback.title,
            review_content=feedback.comment,
        )

        return {"message": "Feedback submitted successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )
