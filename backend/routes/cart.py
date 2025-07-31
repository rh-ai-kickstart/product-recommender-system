from typing import List

from fastapi import APIRouter

from models import CartItem, InteractionType
from services.kafka_service import KafkaService

router = APIRouter()


@router.get("/cart/{user_id}", response_model=List[CartItem])
def get_cart(user_id: str):
    return []


@router.post("/cart", status_code=204)
def add_to_cart(item: CartItem):
    KafkaService().send_interaction(item.user_id, item.product_id, InteractionType.CART)

    # Do the db logic
    return


@router.put("/cart", status_code=204)
def update_cart(item: CartItem):
    return


@router.delete("/cart", status_code=204)
def remove_from_cart(item: CartItem):
    return
