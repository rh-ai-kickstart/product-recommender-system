from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from fastapi.exceptions import HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException
import httpx
import sys
import numpy as np
from routes import auth, products, recommendations, cart, orders, wishlist, feedback, health, interactions, preferences
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

# Custom StaticFiles class for SPA
class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        if len(sys.argv) > 1 and sys.argv[1] == "dev":
            # We are in Dev mode, proxy to the React dev server
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:9000/{path}")
            return Response(response.text, status_code=response.status_code)
        else:
            try:
                return await super().get_response(path, scope)
            except (HTTPException, StarletteHTTPException) as ex:
                if ex.status_code == 404:
                    return await super().get_response("index.html", scope)
                else:
                    raise ex

# Include Routers
# app.include_router(test.router)
app.include_router(preferences.router)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(products.router)
#app.include_router(interactions.router)
app.include_router(recommendations.router)
#app.include_router(cart.router)
#app.include_router(orders.router)
#app.include_router(wishlist.router)
#app.include_router(feedback.router)


# Mount SPA static files at the root - this should be LAST
app.mount("/", SPAStaticFiles(directory="public", html=True), name="spa-static-files")