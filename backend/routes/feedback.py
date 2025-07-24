from fastapi import APIRouter
from models import Feedback, InteractionType
from services.kafka_service import KafkaService

router = APIRouter()


@router.post("/feedback", status_code=204)
def submit_feedback(payload: Feedback):
    KafkaService().send_interaction(
        payload.userId,
        payload.productId,
        InteractionType.RATE,
        payload.rating,
        review_title=payload.title,
        review_content=payload.comment,
    )
    return  # TODO add logic
