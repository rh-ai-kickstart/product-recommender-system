import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from database.models_sql import CartItem as CartItemDB
from database.models_sql import User
from models import CartItem, InteractionType
from services.kafka_service import KafkaService
from routes.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/cart/{user_id}", response_model=List[CartItem])
async def get_cart(
    user_id: str, 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Users can only access their own cart
    if current_user.user_id != user_id:
        # Return empty cart for other users
        return []
    
    # Get cart items from database
    stmt = select(CartItemDB).where(CartItemDB.user_id == user_id)
    result = await db.execute(stmt)
    cart_items = result.scalars().all()

    # Convert to Pydantic models
    return [
        CartItem(
            user_id=item.user_id,
            product_id=item.product_id,
            quantity=item.quantity,
        )
        for item in cart_items
    ]


@router.post("/cart", status_code=204)
async def add_to_cart(
    item: CartItem, 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Users can only add to their own cart
    if current_user.user_id != item.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only add items to your own cart"
        )
    
    KafkaService().send_interaction(
        user_id=item.user_id,
        item_id=item.product_id,
        interaction_type=InteractionType.CART,
    )

    # Check if item already exists in cart
    stmt = select(CartItemDB).where(
        CartItemDB.user_id == item.user_id,
        CartItemDB.product_id == item.product_id,
    )
    result = await db.execute(stmt)
    existing_item = result.scalar_one_or_none()

    if existing_item:
        # Update quantity
        existing_item.quantity += item.quantity or 1
    else:
        # Add new item
        new_item = CartItemDB(
            user_id=item.user_id,
            product_id=item.product_id,
            quantity=item.quantity or 1,
        )
        db.add(new_item)

    await db.commit()


@router.put("/cart", status_code=204)
async def update_cart(
    item: CartItem, 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Users can only update their own cart
    if current_user.user_id != item.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own cart"
        )
    
    # Find existing item
    stmt = select(CartItemDB).where(
        CartItemDB.user_id == item.user_id, CartItemDB.product_id == item.product_id
    )
    result = await db.execute(stmt)
    existing_item = result.scalar_one_or_none()

    if existing_item:
        if item.quantity <= 0:
            # If quantity is 0 or less, delete the item
            await db.delete(existing_item)
            logger.info(
                f"🗑️ Deleted item (quantity 0): user={item.user_id}, product={item.product_id}"
            )
        else:
            # Update quantity
            existing_item.quantity = item.quantity
            logger.info(
                f"📝 Updated quantity: user={item.user_id}, product={item.product_id}, \
                    quantity={item.quantity}"
            )

        await db.commit()
    else:
        logger.info(
            f"⚠️ Item not found for update: user={item.user_id}, \
            product={item.product_id}"
        )

    return


@router.delete("/cart", status_code=204)
async def remove_from_cart(
    item: CartItem, 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Users can only delete from their own cart
    if current_user.user_id != item.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only remove items from your own cart"
        )
    
    # Delete entire item regardless of quantity (for trash button)
    stmt = delete(CartItemDB).where(
        CartItemDB.user_id == item.user_id, CartItemDB.product_id == item.product_id
    )
    result = await db.execute(stmt)
    await db.commit()

    logger.info(
        f"🗑️ Deleted entire item: user={item.user_id}, \
            product={item.product_id}, rows_affected={result.rowcount}"
    )
