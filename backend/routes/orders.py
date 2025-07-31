from datetime import datetime
from typing import List

from fastapi import APIRouter

from models import CheckoutRequest, InteractionType, Order
from services.kafka_service import KafkaService

router = APIRouter()


@router.post("/checkout", response_model=Order)
def checkout(request: CheckoutRequest):
    for item in request.items:
        KafkaService().send_interaction(
            request.user_id,
            item.product_id,
            InteractionType.PURCHASE,
            quantity=item.quantity,
        )

    return Order(
        order_id=1,
        user_id=request.user_id,
        items=request.items,
        total_amount=199.99,
        order_date=datetime.now(),
        status="processing",
    )


@router.get("/orders/{user_id}", response_model=List[Order])
def get_order_history(user_id: str):
    return []
