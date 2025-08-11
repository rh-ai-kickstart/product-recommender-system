from datetime import datetime

import pandas as pd


# Rule-based filtering functions
def _availability_filter(user_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    """Remove items that are out of stock, not released yet, or unavailable in user location.

    Assumptions:
    - Items with arrival_date > current date are not yet available
    - Using 'popular' flag as proxy for stock availability (popular items more likely in stock)
    """
    current_date = datetime.now()

    # Filter out items not yet available and likely out of stock
    available_items_ids_per_user = []
    for _, user in user_df.iterrows():
        available_items_ids = item_df.loc[
            item_df["item_id"].isin(user["top_k_item_ids"])
            & (item_df["arrival_date"] <= current_date)  # Only items already arrived
            & (item_df["popular"] == True),  # Using popular as proxy for availability
            "item_id",
        ]
        available_items_ids_per_user.append(available_items_ids.to_list())
    user_df["top_k_item_ids"] = available_items_ids_per_user
    return user_df


def _demographic_filtering(
    user_df: pd.DataFrame, item_df: pd.DataFrame
) -> pd.DataFrame:
    """Exclude items not suitable for user's age, gender, or region.

    Assumptions:
    - Younger users (18-25) prefer Electronics and Clothing
    - Older users (50+) prefer Home and Books
    - Gender-specific filtering for Clothing category
    """
    filtered_items_per_user = []

    for _, user in user_df.iterrows():
        user_items = user["top_k_item_ids"]
        if len(user_items) == 0:
            continue
        user_items_df = item_df[item_df["item_id"].isin(user_items)]
        # Age-based filtering
        if user["age"] <= 25:
            user_items_df = user_items_df[
                user_items_df["category"].isin(["Electronics", "Clothing", "Sports"])
            ]
        elif user["age"] >= 50:
            user_items_df = user_items_df[
                user_items_df["category"].isin(["Home", "Books"])
            ]

        # Gender-based filtering for Clothing
        if user["gender"] == "M":
            user_items_df = user_items_df[
                ~(
                    (user_items_df["category"] == "Clothing")
                    & (user_items_df["subcategory"] == "Dresses")
                )
            ]
        elif user["gender"] == "F":
            user_items_df = user_items_df[
                ~(
                    (user_items_df["category"] == "Clothing")
                    & (user_items_df["subcategory"].isin(["Shirts", "Pants"]))
                )
            ]

        filtered_items_per_user.append(user_items_df["user_id"].to_list())
    user_df["top_k_item_ids"] = filtered_items_per_user
    return user_df


def _user_history(
    user_df: pd.DataFrame, item_df: pd.DataFrame, interactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Filter out items that the user has already interacted with, if needed.

    Args:
        interactions_df: DataFrame containing user-item interactions
    """
    # Get items each user has interacted with
    interacted_items = interactions_df.groupby("user_id")["item_id"].unique()

    filtered_items = []
    for _, user in user_df.iterrows():
        user_items = item_df.copy()
        if user["user_id"] in interacted_items.index:
            # Exclude items already interacted with
            user_items = user_items[
                ~user_items["item_id"].isin(interacted_items[user["user_id"]])
            ]
        filtered_items.append(user_items.assign(user_id=user["user_id"]))

    return pd.concat(filtered_items) if filtered_items else pd.DataFrame()


def _contextual_filters(user_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    """Session-specific factors like time, day of the week, device type.

    Assumptions:
    - Weekends boost Sports and Home items
    - Morning hours boost Electronics
    - Evening hours boost Books
    """
    current_time = datetime.now()
    is_weekend = current_time.weekday() >= 5  # Saturday = 5, Sunday = 6
    current_hour = current_time.hour

    # Boost certain categories based on context
    if is_weekend:
        item_df = item_df[
            item_df["category"].isin(["Sports", "Home"])
            | (item_df["popular"] == True)  # Keep popular items regardless
        ]
    elif 6 <= current_hour <= 12:  # Morning hours
        item_df = item_df[
            item_df["category"].isin(["Electronics"]) | (item_df["popular"] == True)
        ]
    elif 18 <= current_hour <= 23:  # Evening hours
        item_df = item_df[
            item_df["category"].isin(["Books"]) | (item_df["popular"] == True)
        ]

    return item_df


def filter_items(
    user_df: pd.DataFrame, item_df: pd.DataFrame, interactions_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Filter items for each user by rule-based factors and session factors.

    Args:
        user_df (pd.DataFrame): User features with a suggested user list feature
        item_df (pd.DataFrame): Item features
        interactions_df (pd.DataFrame, optional): User-item interaction history

    Returns:
        pd.DataFrame: Filtered items for each user
    """
    # Apply filters sequentially
    available_items = _availability_filter(user_df, item_df)
    if available_items.empty:
        return pd.DataFrame()

    demographic_filtered = _demographic_filtering(user_df, available_items)
    if demographic_filtered.empty:
        return pd.DataFrame()

    history_filtered = _user_history(user_df, demographic_filtered, interactions_df)
    if history_filtered.empty:
        return pd.DataFrame()

    context_filtered = _contextual_filters(
        user_df, history_filtered.drop(columns=["user_id"])
    )

    # Merge user_ids back and ensure unique items per user
    final_result = (
        history_filtered[["user_id"]]
        .drop_duplicates()
        .merge(context_filtered, how="cross")
    )

    return final_result
