import { Gallery, GalleryItem } from '@patternfly/react-core';
import { ProductCard } from './product-card';
import './Carousel/carousel.css';
import type { ProductData } from '../types';

export const GalleryView = ({ products }: { products: ProductData[] }) => {
  return (
    <div className='gallery-container'>
      <Gallery hasGutter>
        {products.map((product, index) => (
          <GalleryItem key={product.item_id || index}>
            <ProductCard product={product} index={index} />
          </GalleryItem>
        ))}
      </Gallery>
    </div>
  );
};
