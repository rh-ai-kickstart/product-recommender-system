from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db

router = APIRouter()


@router.get("/health/live")
async def liveness_check():
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        return Response(status_code=503)
