import { Gallery, GalleryItem } from '@patternfly/react-core';
import { ProductCard } from './product-card';
import './Carousel/carousel.css';
import type { ProductData } from '../types';

export const GalleryView = ({ products }: { products: ProductData[] }) => {
  return (
    <div className='gallery-container'>
      <Gallery hasGutter>
        {products.map((product, index) => (
          <GalleryItem>
            <ProductCard product={product} index={index} />
          </GalleryItem>
        ))}
      </Gallery>
    </div>
  );
};
