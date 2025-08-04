"""
This script is used for initilizing the backend database.
This should be run by a job once per cluster.
"""

import asyncio
import logging
import subprocess

from database.db import get_engine
from database.fetch_feast_users import seed_users
from database.models_sql import Base

logger = logging.getLogger(__name__)


async def create_tables():
    try:
        async with get_engine().begin() as conn:
            # Drop existing tables (dev only)
            await conn.run_sync(Base.metadata.drop_all)
            # Create fresh schema with updated types
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        raise


async def setup_all():
    try:
        logger.info("🔄 Starting database initialization...")
        await create_tables()
        logger.info("🔄 Seeding users...")
        await seed_users()
        logger.info("✅ Database initialization completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during database initialization: {e}")
        logger.info("🔄 Keeping pod alive for debugging...")
        # Keep the pod running for debugging
        subprocess.run(["tail", "-f", "/dev/null"])


if __name__ == "__main__":
    asyncio.run(setup_all())
