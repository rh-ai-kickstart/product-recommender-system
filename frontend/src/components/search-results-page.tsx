import {
  PageSection,
  Title,
  EmptyState,
  EmptyStateBody,
} from '@patternfly/react-core';
import { GalleryView } from './Gallery';
import { GallerySkeleton } from './gallery-skeleton';
import { useProductSearch } from '../hooks';

interface SearchResultsPageProps {
  query: string;
}

export function SearchResultsPage({ query }: SearchResultsPageProps) {
  const { data, error, isLoading } = useProductSearch(query, query.length > 0);

  const products = data ? data : [];

  if (!query) {
    return (
      <PageSection hasBodyWrapper={false}>
        <EmptyState>
          <Title headingLevel='h4' size='lg'>
            No search query provided
          </Title>
          <EmptyStateBody>
            Please enter a search term to find products.
          </EmptyStateBody>
        </EmptyState>
      </PageSection>
    );
  }

  if (isLoading) {
    return (
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={'h1'} style={{ marginTop: '15px' }}>
          Search Results for "{query}"
        </Title>
        <GallerySkeleton count={8} />
      </PageSection>
    );
  }

  if (error) {
    return (
      <PageSection hasBodyWrapper={false}>
        <EmptyState>
          <Title headingLevel='h4' size='lg'>
            Error searching for products
          </Title>
          <EmptyStateBody>
            There was an error while searching. Please try again.
          </EmptyStateBody>
        </EmptyState>
      </PageSection>
    );
  }

  return (
    <>
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={'h1'} style={{ marginTop: '15px' }}>
          Search Results for "{query}"
        </Title>
        {products.length === 0 ? (
          <EmptyState>
            <Title headingLevel='h4' size='lg'>
              No products found
            </Title>
            <EmptyStateBody>
              No products match your search for "{query}". Try different
              keywords.
            </EmptyStateBody>
          </EmptyState>
        ) : (
          <GalleryView products={products} />
        )}
      </PageSection>
    </>
  );
}
