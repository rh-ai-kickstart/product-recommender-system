# backend/database/create_tables.py
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables before importing engine
env_path = Path(__file__).resolve().parents[2] / "env" / "local.env"
load_dotenv(dotenv_path=env_path)

# Now import the database engine with the correct DATABASE_URL
from database.db import engine
from database.models_sql import Base

async def create_tables():
    async with engine.begin() as conn:
        # Drop existing tables (dev only)
        await conn.run_sync(Base.metadata.drop_all)
        # Create fresh schema with updated types
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    asyncio.run(create_tables())
