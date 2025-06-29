import { Page, PageSection } from '@patternfly/react-core';
import { createFileRoute } from '@tanstack/react-router';
import { Masthead } from '../components/masthead';
import { SimpleSignupPage } from '../components/signup';

export const Route = createFileRoute('/signup')({
  component: Signup,
});

function Signup() {
  return (
    <Page masthead={<Masthead />}>
      <PageSection hasBodyWrapper={false}>
        <SimpleSignupPage />
      </PageSection>
    </Page>
  );
}
