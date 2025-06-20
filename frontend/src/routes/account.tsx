import { Page, PageSection } from '@patternfly/react-core';
import { createFileRoute, redirect } from '@tanstack/react-router';
import { Masthead } from '../components/masthead';
import { AccountPage } from '../components/account-page';

const user = false;

export const Route = createFileRoute('/account')({
  loader: () => {
    if (!user) {
      redirect({
        to: '/login',
        throw: true,
      })
    }
  },
  component: Account,
});

function Account() {
  return (
    <Page masthead={<Masthead />}>
      <PageSection hasBodyWrapper={false}>
        <AccountPage />
      </PageSection>
    </Page>
  );
}
