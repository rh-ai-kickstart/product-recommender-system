from typing import List

from fastapi import APIRouter
from models import Product, WishlistItem

router = APIRouter()


@router.get("/wishlist/{user_id}", response_model=List[Product])
def get_wishlist(user_id: str):
    return []


@router.post("/wishlist", status_code=204)
def add_to_wishlist(item: WishlistItem):
    return


@router.delete("/wishlist", status_code=204)
def remove_from_wishlist(item: WishlistItem):
    return
