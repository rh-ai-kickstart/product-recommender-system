from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from database.models_sql import User
from models import AuthResponse, PreferencesRequest
from models import User as UserResponse
from routes.auth import create_access_token, get_current_user

router = APIRouter(prefix="/users", tags=["users"])


# POST /users/preferences
@router.post(
    "/preferences",
    response_model=AuthResponse,
    status_code=status.HTTP_200_OK,
)
async def set_preferences(
    prefs: PreferencesRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Update DB
    user.preferences = prefs.preferences
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return AuthResponse(
        user=UserResponse(
            user_id=user.user_id,
            email=user.email,
            age=user.age,
            gender=user.gender,
            signup_date=user.signup_date,
            preferences=user.preferences,
            views=[],
        ),
        token=create_access_token(subject=str(user.user_id)),  # Optional: refresh token
    )


# GET /users/preferences
@router.get(
    "/preferences",
    response_model=str,
    status_code=status.HTTP_200_OK,
)
async def get_preferences(user: User = Depends(get_current_user)):
    return user.preferences
