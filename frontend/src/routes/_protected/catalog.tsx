import { Flex, PageSection } from '@patternfly/react-core';
import { createFileRoute } from '@tanstack/react-router';
import { CatalogPage } from '../../components/catalog-page';

export const Route = createFileRoute('/_protected/catalog')({
  component: Catalog,
});

function Catalog() {
  return (
    <PageSection>
      <Flex
        direction={{ default: 'column' }}
        spaceItems={{ default: 'spaceItemsXl' }}
      >
        <CatalogPage />
      </Flex>
    </PageSection>
  );
}
