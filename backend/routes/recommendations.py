from typing import List

from database.models_sql import User
from fastapi import APIRouter, Depends, HTTPException
from models import Product
from models import User as UserSchema  # Pydantic User
from pydantic import BaseModel
from routes.auth import get_current_user  # to resolve JWT user
from services.feast.feast_service import FeastService

router = APIRouter()


# GET for existing users
@router.get("/recommendations/{user_id}", response_model=List[Product])
def get_recommendations(user_id: str):
    try:
        return FeastService().load_items_existing_user(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Input model for new user recommendations
class NewUserRecommendationRequest(BaseModel):
    num_recommendations: int = 10


# POST for new users (authenticated)
@router.post("/recommendations", response_model=List[Product])
async def recommend_for_new_user(
    payload: NewUserRecommendationRequest,
    user: User = Depends(get_current_user),  # SQLAlchemy User
):
    try:
        user_pydantic = UserSchema.model_validate(user)
        return FeastService().load_items_new_user(user_pydantic, k=payload.num_recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
