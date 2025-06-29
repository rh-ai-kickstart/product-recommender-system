import { Page, PageSection } from '@patternfly/react-core';
import { createFileRoute } from '@tanstack/react-router';
import { Masthead } from '../components/masthead';
import { SimpleLoginPage } from '../components/login';

export const Route = createFileRoute('/login')({
  component: Login,
});

function Login() {
  return (
    <Page masthead={<Masthead />}>
      <PageSection hasBodyWrapper={false}>
        <SimpleLoginPage />
      </PageSection>
    </Page>
  );
}
