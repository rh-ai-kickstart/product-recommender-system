"""
This script is used for initilizing the backend database.
This should be run by a job once per cluster.
"""

import asyncio
import logging
import subprocess
import uuid
from collections import deque
from dataclasses import dataclass

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database.db import get_engine
from database.fetch_feast_users import seed_users
from database.models_sql import Base, Category

logger = logging.getLogger(__name__)


@dataclass
class category_dc:
    category_id: uuid
    name: str
    parent_id: uuid


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


async def populate_categories():
    try:
        # Read parquet file containing categories in Category, Parent Category format
        raw_categories_file = "/app/recommendation-core/src/recommendation_core/feature_repo/data/category_relationships.parquet"  # noqa: E501
        df = pd.read_parquet(raw_categories_file)

        engine = get_engine()
        SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

        root_categories_df = df[df["Parent Category"].isnull()]
        root_categories = [
            category_dc(uuid.uuid4(), row["Category"], None)
            for _, row in root_categories_df.iterrows()
        ]
        q = deque(root_categories)

        async with SessionLocal() as session:
            while len(q):
                next_category = q.popleft()
                children_of_next_df = df[df["Parent Category"] == next_category.name]
                children_of_next = [
                    category_dc(uuid.uuid4(), row["Category"], next_category.category_id)
                    for _, row in children_of_next_df.iterrows()
                ]
                q.extend(children_of_next)
                session.add(
                    Category(
                        category_id=next_category.category_id,
                        name=next_category.name,
                        parent_id=next_category.parent_id,
                    )
                )

            await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Unexpected error loading categories in init_backend: {e}")
        raise


async def setup_all():
    try:
        logger.info("🔄 Starting database initialization...")
        await create_tables()
        logger.info("🔄 Seeding users...")
        await seed_users()
        logger.info("✅ Database initialization completed successfully")
        await populate_categories()
        logger.info("✅ Categories populated successfully")
    except Exception as e:
        logger.error(f"❌ Error during database initialization: {e}")
        logger.info("🔄 Keeping pod alive for debugging...")
        # Keep the pod running for debugging
        subprocess.run(["tail", "-f", "/dev/null"])


if __name__ == "__main__":
    asyncio.run(setup_all())
