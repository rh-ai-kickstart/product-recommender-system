import { Page, PageSection } from '@patternfly/react-core';
import { createFileRoute } from '@tanstack/react-router'
import { Masthead } from '../components/masthead';
import { CatalogPage } from '../components/catalog-page';

export const Route = createFileRoute('/catalog')({
  component: Catalog,
});

function Catalog() {
  return (
    <Page masthead={<Masthead />}>
      <PageSection hasBodyWrapper={false}>
        <CatalogPage />
      </PageSection>
    </Page>
  );
}
