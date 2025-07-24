from io import BytesIO
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from models import Product
from PIL import Image, UnidentifiedImageError
from routes.auth import get_current_user  # to resolve JWT user
from services.feast.feast_service import FeastService
from services.kafka_service import KafkaService

router = APIRouter()


@router.get("/products/search", response_model=List[Product])
async def search_products_by_text(query: str, k: int = 5):
    """
    Search products by text query
    """
    try:
        feast = FeastService()
        return feast.search_item_by_text(query, k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/search/image_link", response_model=List[Product])
async def search_products_by_image_link(image_link: str, k: int = 5):
    """
    Search products by image_link
    """
    try:
        feast = FeastService()
        return feast.search_item_by_image_link(image_link, k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/products/search/image", response_model=List[Product])
async def search_products_by_image(image: UploadFile = File(...), k: int = 5):
    """
    Search products by image - Not fully ready yet
    """
    try:
        contents = await image.read()
        try:
            # Try decoding the image in memory
            pil_image = Image.open(BytesIO(contents))
            pil_image.verify()  # Just checks integrity
            pil_image = Image.open(BytesIO(contents))  # Reopen after verify
            print(
                f"Image mode: {pil_image.mode}, size: {pil_image.size}, format: {pil_image.format}"
            )
        except UnidentifiedImageError as e:
            print(f"[ImageError] Cannot identify image: {e}")
            raise HTTPException(status_code=400, detail="Unsupported or invalid image format.")
        except Exception as e:
            print(f"[ImageError] General image load error: {e}")
            raise HTTPException(status_code=400, detail="Failed to process uploaded image.")

        feast = FeastService()
        return feast.search_item_by_image_file(pil_image, k)

    except HTTPException:
        raise  # Pass through
    except Exception as e:
        print(f"[InternalError] {e}")
        raise HTTPException(status_code=500, detail="Unexpected server error during image search.")


@router.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: str, user_id=Depends(get_current_user)):
    """
    Get product details by ID
    """
    # Send view interaction to Kafka
    KafkaService().send_interaction(
        user_id=user_id, item_id=product_id, interaction_type="negative_view"
    )

    try:
        feast = FeastService()
        return feast.get_item_by_id(product_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/products/{product_id}/interactions/click", status_code=204)
async def record_product_click(product_id: str, user_id=Depends(get_current_user)):
    """
    Records a product click interaction event
    """
    KafkaService().send_interaction(
        user_id=user_id, item_id=product_id, interaction_type="positive_view"
    )
    return
