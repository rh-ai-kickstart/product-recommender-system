from datetime import datetime

from sqlalchemy import Date, DateTime, Float, Integer, String, Text, ForeignKey
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship
from sqlalchemy_utils import UUIDType
import uuid

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(27), primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password: Mapped[str] = mapped_column(
        String, nullable=True
    )  # raw (used only for mock data/gen)
    hashed_password: Mapped[str] = mapped_column(String, nullable=True)
    age: Mapped[int] = mapped_column(Integer, nullable=True)
    gender: Mapped[str] = mapped_column(String, nullable=True)
    signup_date: Mapped[Date] = mapped_column(Date, nullable=True)
    preferences: Mapped[str] = mapped_column(String, nullable=True)
    user_preferences : Mapped[list["UserPreference"]] = relationship(
        "UserPreference",
        back_populates="user"
    )


class CartItem(Base):
    __tablename__ = "cart_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(27), index=True)
    product_id: Mapped[str] = mapped_column(String, index=True)
    quantity: Mapped[int] = mapped_column(Integer)


class StreamInteraction(Base):
    __tablename__ = "stream_interaction"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(27), index=True)
    item_id: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    interaction_type: Mapped[str] = mapped_column(String)
    rating: Mapped[float] = mapped_column(Float, nullable=True)
    quantity: Mapped[float] = mapped_column(Float, nullable=True)
    review_title: Mapped[str] = mapped_column(Text, nullable=True)
    review_content: Mapped[str] = mapped_column(Text, nullable=True)
    interaction_id: Mapped[str] = mapped_column(String, unique=True, index=True)


class Category(Base):
    __tablename__ = "category"
    category_id: Mapped[uuid.UUID] = mapped_column(UUIDType, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String)
    parent_id: Mapped[uuid.UUID] = mapped_column(UUIDType, ForeignKey('category.category_id'),
                                                 nullable=True)

    parent: Mapped["Category"] = relationship(
        "Category",
        remote_side="Category.category_id",
        back_populates="sub_categories",
        foreign_keys="Category.parent_id"
    )

    sub_categories: Mapped[list["Category"]] = relationship(
        "Category",
        back_populates="parent",
        lazy="dynamic",
        foreign_keys="Category.parent_id"
    )

    products: Mapped[list["Product"]] = relationship(
        "Product",
        back_populates="category"
    )


class Product(Base):
    __tablename__ = "products"
    item_id: Mapped[uuid.UUID] = mapped_column(UUIDType, primary_key=True, default=uuid.uuid4)
    category_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("category.category_id"))
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    price: Mapped[float] = mapped_column(Float)
    avg_rating: Mapped[float] = mapped_column(Float)
    num_ratings: Mapped[int] = mapped_column(Integer)
    popular: Mapped[float] = mapped_column(Float)
    new_arrival: Mapped[float] = mapped_column(Float)
    on_sale: Mapped[float] = mapped_column(Float)
    arrival_date: Mapped[Date] = mapped_column(Date)
    category: Mapped["Category"] = relationship("Category", back_populates="products")


class UserPreference(Base):
    __tablename__ = "user_preferences"
    id: Mapped[uuid.UUID] = mapped_column(UUIDType, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(27), ForeignKey("users.user_id"))
    category_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("category.category_id"))

    user: Mapped["User"] = relationship(
        "User",
        back_populates="user_preferences"
    )

    category: Mapped["Category"] = relationship(
        "Category"
    )


# class Feedback(Base):
#    __tablename__ = "feedback"
#    id: Mapped[int] = mapped_column(Integer, primary_key=True)
#    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
#    product_id: Mapped[int] = mapped_column(ForeignKey("products.item_id"))
#    rating: Mapped[float] = mapped_column(Float)
#    comment: Mapped[str] = mapped_column(String, nullable=True)

# class Order(Base):
#    __tablename__ = "orders"
#    order_id: Mapped[int] = mapped_column(Integer, primary_key=True)
#    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
#    total_amount: Mapped[float] = mapped_column(Float)
#    order_date: Mapped[datetime.datetime] = mapped_column(DateTime)
#    status: Mapped[str] = mapped_column(String)

# class WishlistItem(Base):
#    __tablename__ = "wishlist"
#    id: Mapped[int] = mapped_column(Integer, primary_key=True)
#    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
#    product_id: Mapped[int] = mapped_column(ForeignKey("products.item_id"))


# class Interactions(Base):
#    __tablename__ = "interactions"
#    id: Mapped[int] = mapped_column(Integer, primary_key=True)
#    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
#    product_id: Mapped[int] = mapped_column(ForeignKey("products.item_id"))
#    rating: Mapped[float] = mapped_column(Float)
#    quantity: Mapped[int] = mapped_column(Integer)

# class NegInteractions(Base):
#    __tablename__ = "neg_interactions"
#    id: Mapped[int] = mapped_column(Integer, primary_key=True)
#    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
#    product_id: Mapped[int] = mapped_column(ForeignKey("products.item_id"))
#    rating: Mapped[float] = mapped_column(Float)
