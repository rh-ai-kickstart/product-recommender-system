import argparse
import logging
import pathlib
import random
import secrets
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

categories = [
    "Computers&Accessories|Accessories&Peripherals|Cables&Accessories|Cables|USBCables",
    "Electronics|WearableTechnology|SmartWatches",
    "Electronics|Mobiles&Accessories|Smartphones&BasicMobiles|Smartphones",
    "Electronics|HomeTheater,TV&Video|Televisions|SmartTelevisions",
    "Electronics|Headphones,Earbuds&Accessories|Headphones|In-Ear",
    "Electronics|HomeTheater,TV&Video|Accessories|RemoteControls",
    "Home&Kitchen|Kitchen&HomeAppliances|SmallKitchenAppliances|MixerGrinders",
    "Computers&Accessories|Accessories&Peripherals|Keyboards,Mice&InputDevices|Mice",
    "Home&Kitchen|Kitchen&HomeAppliances|Vacuum,Cleaning&Ironing|Irons,Steamers&Accessories|Irons|DryIrons",
    "Electronics|HomeTheater,TV&Video|Accessories|Cables|HDMICables",
    "Home&Kitchen|Heating,Cooling&AirQuality|WaterHeaters&Geysers|InstantWaterHeaters",
    "Home&Kitchen|Kitchen&HomeAppliances|Vacuum,Cleaning&Ironing|Irons,Steamers&Accessories|LintShavers",
    "Home&Kitchen|Kitchen&HomeAppliances|SmallKitchenAppliances|Kettles&HotWaterDispensers|ElectricKettles",
    "Home&Kitchen|Kitchen&HomeAppliances|SmallKitchenAppliances|HandBlenders",
    "Home&Kitchen|Heating,Cooling&AirQuality|RoomHeaters|FanHeaters",
    "Home&Kitchen|Heating,Cooling&AirQuality|RoomHeaters|ElectricHeaters",
    "Computers&Accessories|NetworkingDevices|NetworkAdapters|WirelessUSBAdapters",
    "Electronics|Mobiles&Accessories|MobileAccessories|Chargers|WallChargers",
    "Computers&Accessories|Accessories&Peripherals|LaptopAccessories|Lapdesks",
    "Electronics|Accessories|MemoryCards|MicroSD",
    "Home&Kitchen|Kitchen&HomeAppliances|SmallKitchenAppliances|Kettles&HotWaterDispensers|Kettle&ToasterSets",
    "Home&Kitchen|HomeStorage&Organization|LaundryOrganization|LaundryBaskets",
    "Home&Kitchen|Kitchen&HomeAppliances|WaterPurifiers&Accessories|WaterFilters&Purifiers",
    "Electronics|Mobiles&Accessories|MobileAccessories|Chargers|PowerBanks",
    "Home&Kitchen|Heating,Cooling&AirQuality|WaterHeaters&Geysers|StorageWaterHeaters",
    "Home&Kitchen|Kitchen&HomeAppliances|Vacuum,Cleaning&Ironing|Irons,Steamers&Accessories|Irons|SteamIrons",
    "Home&Kitchen|Kitchen&HomeAppliances|SmallKitchenAppliances|JuicerMixerGrinders",
    "Computers&Accessories|Accessories&Peripherals|Keyboards,Mice&InputDevices|GraphicTablets",
    "Home&Kitchen|Heating,Cooling&AirQuality|Fans|CeilingFans",
    "Home&Kitchen|Kitchen&HomeAppliances|SmallKitchenAppliances|EggBoilers",
    "Home&Kitchen|Kitchen&HomeAppliances|SmallKitchenAppliances|SandwichMakers",
    "Home&Kitchen|Kitchen&HomeAppliances|WaterPurifiers&Accessories|WaterPurifierAccessories",
    "Electronics|Mobiles&Accessories|MobileAccessories|Stands",
    "Computers&Accessories|ExternalDevices&DataStorage|PenDrives",
    "Computers&Accessories|Accessories&Peripherals|Keyboards,Mice&InputDevices|Keyboard&MouseSets",
    "Home&Kitchen|Kitchen&HomeAppliances|SmallKitchenAppliances|InductionCooktop",
    "Home&Kitchen|Kitchen&HomeAppliances|SmallKitchenAppliances|DigitalKitchenScales",
    "Computers&Accessories|NetworkingDevices|Routers",
    "Electronics|Mobiles&Accessories|Smartphones&BasicMobiles|BasicMobiles",
    "Home&Kitchen|Heating,Cooling&AirQuality|WaterHeaters&Geysers|ImmersionRods",
    "Electronics|Headphones,Earbuds&Accessories|Headphones|On-Ear",
    "Computers&Accessories|Accessories&Peripherals|Keyboards,Mice&InputDevices|Keyboard&MiceAccessories|MousePads",
    "Home&Kitchen|Kitchen&HomeAppliances|Vacuum,Cleaning&Ironing|Vacuums&FloorCare|Vacuums|HandheldVacuums",
    "Electronics|Mobiles&Accessories|MobileAccessories|StylusPens",
    "Home&Kitchen|Kitchen&HomeAppliances|SmallKitchenAppliances|MiniFoodProcessors&Choppers",
    "Electronics|Mobiles&Accessories|MobileAccessories|Maintenance,Upkeep&Repairs|ScreenProtectors",
    "OfficeProducts|OfficePaperProducts|Paper|Stationery|Notebooks,WritingPads&Diaries|CompositionNotebooks",
    "Electronics|GeneralPurposeBatteries&BatteryChargers|DisposableBatteries",
    "Home&Kitchen|Kitchen&HomeAppliances|SmallKitchenAppliances|VacuumSealers",
    "Computers&Accessories|Accessories&Peripherals|LaptopAccessories|Bags&Sleeves|LaptopSleeves&Slipcases",
]
unique_categories = list(
    set(
        [item for sublist in [cat.split("|") for cat in categories] for item in sublist]
    )
)
# Set random seed for reproducibility
np.random.seed(42)


def generate_id(length=26):
    characters = string.ascii_uppercase + string.digits
    user_id = "".join(secrets.choice(characters) for _ in range(length))
    return user_id


# Generate user data
def generate_users(num_users, from_id=0):
    users = []
    for user_id in [generate_id() for i in range(num_users)]:
        signup_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))

        # Generate user preferences (categories they tend to like)
        preferences = "|".join(random.sample(unique_categories, random.randint(1, 5)))

        users.append(
            {
                "user_id": user_id,
                "user_name": "Customer",
                "signup_date": signup_date,
                "preferences": preferences,
            }
        )

    return pd.DataFrame(users)


def generate_items():
    items = []
    # Define categories and subcategories based on sample data

    item_df = pd.read_csv("src/recommendation_core/feature_repo/data/generated_amazon.csv")
    for item_id, item in item_df.iterrows():
        category = item.category

        # Generate prices and discount
        actual_price = np.round(np.random.uniform(500, 2000), 2)
        discount_percentage = np.round(np.random.uniform(0.1, 0.8), 2)
        discounted_price = np.round(actual_price * (1 - discount_percentage), 2)

        # Generate rating and rating count
        rating = np.round(np.random.uniform(3.0, 5.0), 1)
        rating_count = np.random.randint(1000, 8000)

        # Generate product ID
        item_id = f"B0{np.random.randint(10000000, 99999999)}"

        # Generate product name and description
        # Generate product name using Ollama API
        product_name = item.item_name
        about_product = item.item_description

        # Generate URLs
        img_link = f"https://raw.githubusercontent.com/rh-ai-kickstart/product-recommender-system/main/recommendation-core/generation/data/generated_images/item_{product_name.replace(' ', '%20')}.png"
        product_link = (
            f"https://www.amazon.in/Wayona-Braided-WN{np.random.randint(1000, 9999)}..."
        )

        arrival_date = datetime(2023, 1, 1) + timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
        )
        items.append(
            {
                "item_id": item_id,
                "product_name": product_name,
                "category": category,
                "discounted_price": discounted_price,
                "actual_price": actual_price,
                "discount_percentage": discount_percentage,
                "rating": rating,
                "rating_count": rating_count,
                "about_product": about_product,
                "arrival_date": arrival_date,
                "img_link": img_link,
                "product_link": product_link,
            }
        )

    return pd.DataFrame(items)


# Generate interactions between users and items
def generate_interactions(
    users_df: pd.DataFrame, items_df: pd.DataFrame, num_interactions: int
):
    interactions = []

    # Ensure we have sufficient users and items
    user_ids = users_df["user_id"].tolist()
    item_ids = items_df["item_id"].tolist()
    items_df = items_df.copy()
    items_df["category"] = items_df.category.str.split("|")
    items_df.explode(["category"])

    for _ in range(num_interactions):
        user_id = random.sample(user_ids, 1)[0]

        # Users are more likely to interact with items in their preferred categories
        user_prefs = (
            users_df.loc[users_df["user_id"] == user_id, "preferences"]
            .iloc[0]
            .split("|")
        )

        # Biased item selection based on user preferences
        if (
            np.random.random() < 0.9 and user_prefs
        ):  # 70% chance to select from preferred categories
            preferred_items = items_df[items_df["category"].isin(user_prefs)]
            if not preferred_items.empty:
                item = preferred_items.sample(1).iloc[0]
                item_id = item["item_id"]
            else:
                item_id = random.sample(item_ids, 1)[0]
        else:
            item_id = random.sample(item_ids, 1)[0]

        # Generate interaction details
        timestamp = datetime(2024, 1, 1) + timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
        )

        # Different types of interactions
        interaction_type = np.random.choice(
            ["positive_view", "negative_view", "cart", "purchase", "rate"],
            p=[0.1, 0.6, 0.1, 0.15, 0.05],
        )

        # Additional data based on interaction type
        if interaction_type == "rate":
            rating = float(np.random.randint(3, 6))  # 1-5 rating
            review_title = "Good" if rating >= 3 else "Bad"
            review_content = "So good" if review_title == "Good" else "Dont buy it"
        else:
            rating = None
            review_title = "No review"
            review_content = "no content"

        if interaction_type == "purchase":
            quantity = float(np.random.randint(1, 4))
        else:
            quantity = None

        interactions.append(
            {
                "interaction_id": generate_id(),
                "user_id": user_id,
                "item_id": item_id,
                "timestamp": timestamp,
                "interaction_type": interaction_type,
                "review_title": review_title,
                "review_content": review_content,
                "rating": rating,
                "quantity": quantity,
            }
        )

    return pd.DataFrame(interactions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Recommendation System dataset"
    )
    parser.add_argument("--n_users", help="Number of users", type=int, default=1000)
    parser.add_argument("--n_items", type=int, help="Number of items", default=5000)
    parser.add_argument(
        "--n_interactions",
        help="Number of interactions of users and items",
        default=20000,
        type=int,
    )

    args = parser.parse_args()
    # Generate the datasets
    users = generate_users(args.n_users)
    items = generate_items()
    interactions = generate_interactions(users, items, args.n_interactions)

    # Display sample of each dataset
    logger.info("Users sample:")
    logger.info(users.head())
    logger.info("\nItems sample:")
    logger.info(items.head())
    logger.info("\nInteractions sample:")
    logger.info(interactions.head())

    data_path = pathlib.Path("src/recommendation_core/feature_repo/data")
    data_path.mkdir(parents=True, exist_ok=True)

    # Save to parquet files
    users.to_parquet(data_path / "recommendation_users.parquet", index=False)
    items.to_parquet(data_path / "recommendation_items.parquet", index=False)
    interactions.to_parquet(
        data_path / "recommendation_interactions.parquet", index=False
    )

    # Create dummy dataframes for push source
    dummy_item_embed_df = pd.DataFrame(
        columns=["item_id", "embedding", "event_timestamp"],
        data=[[generate_id(), [1.0, 2.0], datetime.now() + timedelta(days=365 * 7)]],
    )
    dummy_user_items_df = pd.DataFrame(
        columns=["user_id", "top_k_item_ids", "event_timestamp"],
        data=[
            [
                generate_id(),
                [generate_id(), generate_id()],
                datetime.now() + timedelta(days=365 * 7),
            ]
        ],
    )
    dummy_user_embed_df = pd.DataFrame(
        columns=["user_id", "embedding", "event_timestamp"],
        data=[[generate_id(), [1.0, 2.0], datetime.now() + timedelta(days=365 * 7)]],
    )
    # Dummy textual / clip features for push source
    dummy_textual_feature_df = pd.DataFrame(
        columns=["item_id", "about_product_embedding", "event_timestamp"],
        data=[[generate_id(), [1.0, 2.0], datetime.now() + timedelta(days=365 * 7)]],
    )
    dummy_clip_feature_df = pd.DataFrame(
        columns=["item_id", "clip_latent_space_embedding", "event_timestamp"],
        data=[[generate_id(), [1.0, 2.0], datetime.now() + timedelta(days=365 * 7)]],
    )

    dummy_item_embed_df.to_parquet(
        data_path / "dummy_item_embed.parquet", index=False
    )
    dummy_user_embed_df.to_parquet(
        data_path / "dummy_user_embed.parquet", index=False
    )
    dummy_user_items_df.to_parquet(data_path / "user_items.parquet", index=False)
    dummy_textual_feature_df.to_parquet(
        data_path / "item_textual_features_embed.parquet", index=False
    )
    dummy_clip_feature_df.to_parquet(
        data_path / "item_clip_features_embed.parquet", index=False
    )
