import { Page, PageSection } from '@patternfly/react-core';
import { createFileRoute } from '@tanstack/react-router';
import { Masthead } from '../components/masthead';
import { LandingPage } from '../components/landing-page';

export const Route = createFileRoute('/')({
  component: Recommendations,
});

const pageId = 'primary-app-container';

function Recommendations() {
  return (
    <Page mainContainerId={pageId} masthead={<Masthead />}>
      <PageSection hasBodyWrapper={false}>
        <LandingPage />
      </PageSection>
    </Page>
  );
}
