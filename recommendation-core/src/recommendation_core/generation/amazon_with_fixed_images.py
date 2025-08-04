import logging
import random
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def fix_url(url: str):
    index_1 = url.find("/images/W/")
    if index_1 == -1:
        return url, False

    index_2 = url.find("/images/", index_1+1)
    if index_2 == -1:
        return url, False

    url = url[:index_1] + url[index_2:]
    return url, True


def main():
    random.seed(739)
    base = Path(__file__).parent.parent / 'src' / 'feature_repo' / 'data'
    item_df = pd.read_csv(base / 'amazon.csv')
    total = 0
    fixed = 0

    def function(row):
        img_link, changed = fix_url(row['img_link'])
        nonlocal total
        nonlocal fixed
        total = total + 1
        if changed:
            fixed = fixed + 1
        return img_link

    item_df['img_link'] = item_df.apply(function, axis=1)
    item_df.to_csv(base / 'amazon_with_fixed_images.csv', index=False)
    logger.info("fixed images:", fixed, " / ", total)


if __name__ == '__main__':
    main()
