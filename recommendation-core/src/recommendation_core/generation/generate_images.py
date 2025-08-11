import logging
from pathlib import Path
from pprint import pprint

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)


def main():
    data_directory = Path(__file__).parent.joinpath("data")
    item_df = pd.read_parquet(
        "src/recommendation_core/feature_repo/data/item_df_output.parquet"
    )
    pprint(item_df)

    # Load the Stable Diffusion pipeline (open model)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Directory to save images
    output_dir = data_directory.joinpath("generated_images")
    output_dir.mkdir(exist_ok=True)

    # Generate images for each row
    for idx, row in item_df.iterrows():
        product_name = row["product_name"]
        prompt = row["about_product"]  # Change to your prompt column if needed
        image = pipe(prompt).images[0]
        image.save(output_dir / f"item_{product_name}.png")
        logger.info(f"Generated image for item {product_name}")


if __name__ == "__main__":
    main()
