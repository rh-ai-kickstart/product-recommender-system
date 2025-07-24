import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth, health, preferences, products, recommendations

# from routes import test

app = FastAPI()

# Set random seed for reproducibility
np.random.seed(42)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
# app.include_router(test.router)
app.include_router(preferences.router)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(products.router)
# app.include_router(interactions.router)
app.include_router(recommendations.router)
# app.include_router(cart.router)
# app.include_router(orders.router)
# app.include_router(wishlist.router)
# app.include_router(feedback.router)

categories = ["Electronics", "Books", "Clothing", "Home", "Sports"]
subcategories = {
    "Electronics": ["Smartphones", "Laptops", "Cameras", "Audio", "Accessories"],
    "Books": ["Fiction", "Non-fiction", "Science", "History", "Self-help"],
    "Clothing": ["Shirts", "Pants", "Dresses", "Shoes", "Accessories"],
    "Home": ["Kitchen", "Furniture", "Decor", "Bedding", "Appliances"],
    "Sports": ["Fitness", "Outdoor", "Team Sports", "Footwear", "Equipment"],
}


# Mock data
mock_products = [
    {"id": 1, "name": "Smartphone", "category": "Tech", "rating": 4.5},
    {"id": 2, "name": "Yoga Mat", "category": "Fitness", "rating": 4.7},
    {"id": 3, "name": "Espresso Maker", "category": "Home", "rating": 4.3},
]

user_views = {1: [mock_products[0], mock_products[2]]}


"""
@app.get("/recommendations")
def get_recommendations(userId: Optional[int] = None):
    print("recommendations called")
    return mock_products

@app.get("/product/{id}")
def get_product(id: int):
    prod = next((p for p in mock_products if p["id"] == id), None)
    if prod:
        return {**prod, "description": "This is a detailed product description."}
    raise HTTPException(status_code=404, detail="Product not found")

@app.get("/search")
def search_products(q: str):
    print("search called")
    results = [p for p in mock_products if q.lower() in p["name"].lower()]
    return results

@app.get("/history")
def get_history(userId: int):
    return user_views.get(userId, [])

@app.post("/feedback")
def post_feedback(feedback: Feedback):
    feedbacks.append(feedback.dict())
    return {"message": "Feedback received"}

@app.post("/login")
def login(login_req: LoginRequest):
    print("login called")
    if login_req.email == "user@example.com" and login_req.password == "1234":
        return {"user": {"id": 1, "email": login_req.email, "name": "Sample User"}}
    raise HTTPException(status_code=401, detail="Invalid credentials")
"""
