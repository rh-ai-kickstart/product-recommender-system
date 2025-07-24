"""
This script is used for initilizing the backend database.
This should be run by a job once per cluster.
"""

import asyncio

from database.db import get_engine
from database.fetch_feast_users import seed_users
from database.models_sql import Base


async def create_tables():
    async with get_engine().begin() as conn:
        # Drop existing tables (dev only)
        await conn.run_sync(Base.metadata.drop_all)
        # Create fresh schema with updated types
        await conn.run_sync(Base.metadata.create_all)


async def setup_all():
    await create_tables()
    await seed_users()


if __name__ == "__main__":
    asyncio.run(setup_all())
