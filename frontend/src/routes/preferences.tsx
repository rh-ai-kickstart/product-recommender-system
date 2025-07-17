import { Page, PageSection } from "@patternfly/react-core";
import { createFileRoute } from "@tanstack/react-router";
import { Masthead } from "../components/masthead";
import { PreferencePage } from "../components/preferences";

export const Route = createFileRoute("/preferences")({
  component: Preferences,
});

function Preferences() {
  return (
    <Page masthead={<Masthead />}>
      <PageSection>
        <PreferencePage />
      </PageSection>
    </Page>
  );
}
