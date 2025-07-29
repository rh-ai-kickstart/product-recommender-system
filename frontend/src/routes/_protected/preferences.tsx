import { createFileRoute } from '@tanstack/react-router';
import { PageSection } from '@patternfly/react-core';
import { PreferencePage } from '../../components/preferences';

export const Route = createFileRoute('/_protected/preferences')({
  component: PreferencesComponent,
});

function PreferencesComponent() {
  return (
    <PageSection>
      <PreferencePage />
    </PageSection>
  );
}
