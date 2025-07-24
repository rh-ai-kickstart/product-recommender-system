import { Gallery, GalleryItem } from '@patternfly/react-core';
import { ProductCardSkeleton } from './product-card-skeleton';
import './Carousel/carousel.css';

interface GallerySkeletonProps {
  count?: number; // Number of skeleton cards to display, defaults to 12
}

export const GallerySkeleton: React.FunctionComponent<GallerySkeletonProps> = ({
  count = 12,
}) => {
  // Create an array of indices to map over
  const skeletonIndices = Array.from({ length: count }, (_, i) => i);

  return (
    <div className='gallery-container'>
      <Gallery hasGutter>
        {skeletonIndices.map(index => (
          <GalleryItem
            className='cards-container'
            style={{
              marginTop: '15px',
              overflow: 'hidden',
            }}
            key={`skeleton-${index}`}
          >
            <ProductCardSkeleton index={index} />
          </GalleryItem>
        ))}
      </Gallery>
    </div>
  );
};
