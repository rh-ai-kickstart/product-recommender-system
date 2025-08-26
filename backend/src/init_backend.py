"""
This script is used for initilizing the backend database.
This should be run by a job once per cluster.
"""

import asyncio
import logging
import subprocess

from database.db import get_engine, get_db
from database.fetch_feast_users import seed_users
from database.models_sql import Base, Category

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


async def populate_categories():
    try:
        async with get_db() as session:
            # Create a parent category
            parent_category_name = "Electronics"
            parent_category = Category(name=parent_category_name)
            session.add(parent_category)
            session.commit()
            
            # Assert it was created successfully
            retrieved_parent = session.get(Category, parent_category.category_id)
            assert retrieved_parent is not None
            assert retrieved_parent.name == parent_category_name
            print(f"✅ Successfully created parent category: {retrieved_parent.name}")

            # Create a sub-category with a parent
            sub_category_name = "Laptops"
            sub_category = Category(name=sub_category_name, parent_id=parent_category.category_id)
            session.add(sub_category)
            session.commit()

            # Assert the sub-category was created and has the correct parent
            retrieved_sub = session.get(Category, sub_category.category_id)
            assert retrieved_sub is not None
            assert retrieved_sub.name == sub_category_name
            assert retrieved_sub.parent_id == parent_category.category_id
            assert retrieved_sub.parent.name == parent_category_name
            print(f"✅ Successfully created sub-category: {retrieved_sub.name} under {retrieved_sub.parent.name}")

    except Exception as e:
        logger.error(f"❌ Error populating categories: {e}")
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
