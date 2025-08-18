from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from models import CheckoutRequest, InteractionType, Order
from services.database_service import db_service  # Use global instance

router = APIRouter()


@router.post("/checkout", response_model=Order)
async def checkout(request: CheckoutRequest, db: AsyncSession = Depends(get_db)):
    # Log purchase interactions to database (replaces Kafka)
    for item in request.items:
        await db_service.log_interaction(
            db=db,
            user_id=request.user_id,
            item_id=item.product_id,
            interaction_type=InteractionType.PURCHASE,
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
