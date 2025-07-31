import { PageSection, Title } from '@patternfly/react-core';
import { GalleryView } from './Gallery';
import { GallerySkeleton } from './gallery-skeleton';
import { usePersonalizedRecommendations } from '../hooks/useRecommendations';

export function CatalogPage() {
  const { data, isLoading } = usePersonalizedRecommendations();

  const products = data ?? [];

  return (
    <>
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={'h1'} style={{ marginTop: '15px' }}>
          Catalog
        </Title>
        {isLoading ? (
          <GallerySkeleton count={12} />
        ) : (
          <GalleryView products={products} />
        )}
      </PageSection>
    </>
  );
}
