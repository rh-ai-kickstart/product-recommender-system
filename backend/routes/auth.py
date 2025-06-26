from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import random, string
from datetime import date
from feast import FeatureStore
from models import SignUpRequest, LoginRequest, AuthResponse, User as UserResponse
from database.models_sql import User, Login
from database.db import get_db
from services.kafka_service import kafka_service  # Kafka send (re-enable when broker is available)
from services.security import (
    hash_password,
    verify_password,
    create_access_token,
    SECRET_KEY,
    ALGORITHM,
)

# OAuth2 scheme for Bearer token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Utility: generate a 26-digit integer ID starting with 1122
def generate_user_id() -> int:
    suffix = ''.join(random.choices(string.digits, k=22))
    return int("1122" + suffix)

# Dependency: get current user from JWT
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            raise credentials_exception
        user_id = int(sub)
    except (JWTError, ValueError):
        raise credentials_exception
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise credentials_exception
    return user

# Request model for setting preferences
class PreferencesRequest(BaseModel):
    preferences: str

class ExistingLoginRequest(BaseModel):
    user_id: int

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post(
    "/signup",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
)
async def signup(
    payload: SignUpRequest,
    db: AsyncSession = Depends(get_db),
):
    # Prevent duplicates
    result = await db.execute(select(Login).where(Login.email == payload.email))
    if result.scalar_one_or_none():
        raise HTTPException(400, "Email already registered")

    # Create user with generated ID
    new_id = generate_user_id()
    user = User(
        user_id=new_id,
        email=payload.email,
        age=payload.age,
        gender=payload.gender,
        signup_date=date.today(),
        preferences="",
    )
    db.add(user)
    await db.flush()

    # Store credentials
    login_entry = Login(
        user_id=user.user_id,
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    db.add(login_entry)
    await db.commit()
    await db.refresh(user)

    # Issue JWT
    token = create_access_token(subject=str(user.user_id))

    user_response = UserResponse(
        user_id=user.user_id,
        email=user.email,
        age=user.age,
        gender=user.gender,
        signup_date=user.signup_date,
        preferences=user.preferences,
        views=[],
    )
    return AuthResponse(user=user_response, token=token)


@router.post(
    "/login",
    response_model=AuthResponse,
)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    # Check credentials
    result = await db.execute(select(Login).where(Login.email == payload.email))
    login_entry = result.scalar_one_or_none()
    if not login_entry or not verify_password(payload.password, login_entry.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    result = await db.execute(select(User).where(User.user_id == login_entry.user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(500, "User record missing")

    token = create_access_token(subject=str(user.user_id))

    user_response = UserResponse(
        user_id=user.user_id,
        email=user.email,
        age=user.age,
        gender=user.gender,
        signup_date=user.signup_date,
        preferences=user.preferences,
        views=[],
    )
    return AuthResponse(user=user_response, token=token)

@router.post(
    "/existing-login",
    response_model=AuthResponse,
)
async def existing_login(
    payload: ExistingLoginRequest,
):
    user_id = payload.user_id

    # Ensure this is NOT a 1122-prefixed ID
    if str(user_id).startswith("1122"):
        raise HTTPException(status_code=400, detail="This is not an existing user ID")

    # Attempt to fetch from Feast
    try:
        store = FeatureStore(repo_path="feature_repo")  # Adjust path if needed
        features = store.get_online_features(
            features=[
                "user_features:age",
                "user_features:gender",
                "user_features:preferences"
            ],
            entity_rows=[{"user_id": user_id}]
        ).to_dict()
    except Exception:
        raise HTTPException(status_code=500, detail="Error connecting to Feast")

    # Confirm user exists in feature store
    if not features.get("age") or features["age"][0] is None:
        raise HTTPException(status_code=401, detail="User not found in feature store")

    # Construct user-like object
    user_response = UserResponse(
        user_id=user_id,
        email="",  # Unknown for existing Feast users
        age=features["age"][0],
        gender=features["gender"][0],
        signup_date=date.today(),  # Could default or skip
        preferences=features["preferences"][0] or "",
        views=[],
    )

    token = create_access_token(subject=str(user_id))

    return AuthResponse(user=user_response, token=token)


@router.post(
    "/preferences",
    response_model=AuthResponse,
)
async def set_preferences(
    prefs: PreferencesRequest,
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme),
):
    # Authenticate
    user = await get_current_user(token, db)

    # Update preferences in DB
    user.preferences = prefs.preferences
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Send Kafka event after preferences chosen (commented until broker ready)
  #  kafka_service.send_new_user(
   #         user_id=user.user_id,
    #        user_name=user.email,
     #       preferences=user.preferences,
      #  )

    # Issue a fresh token (or reuse existing)
    new_token = token

    user_response = UserResponse(
        user_id=user.user_id,
        email=user.email,
        age=user.age,
        gender=user.gender,
        signup_date=user.signup_date,
        preferences=user.preferences,
        views=[],
    )
    return AuthResponse(user=user_response, token=new_token)
