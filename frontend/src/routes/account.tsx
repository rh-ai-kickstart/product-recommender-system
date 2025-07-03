import {
  Page,
  PageSection,
  TreeView,
  type TreeViewDataItem,
  Sidebar,
  SidebarPanel,
  SidebarContent,
  Flex,
} from "@patternfly/react-core";
import { createFileRoute, redirect } from "@tanstack/react-router";
import { Masthead } from "../components/masthead";
import { Wishlist } from "../components/wishlist";

const user = true;

export const Route = createFileRoute("/account")({
  loader: () => {
    if (!user) {
      redirect({
        to: "/login",
        throw: true,
      });
    }
  },
  component: Account,
});

const options: TreeViewDataItem[] = [
  {
    name: "My Account",
    id: "account",

    children: [
      {
        name: "Wishlist",
        id: "saved-products",
      },
    ],
    defaultExpanded: true,
  },
];

function Account() {
  return (
    <Page masthead={<Masthead />}>
      <Sidebar>
        <SidebarPanel variant="sticky">
          <TreeView
            aria-label="account navigation"
            data={options}
            allExpanded={true}
            hasGuides={true}
            // onSelect={onSelect}
          />
        </SidebarPanel>
        <SidebarContent>
          <PageSection>
            <Flex
              direction={{ default: "column" }}
              spaceItems={{ default: "spaceItemsMd" }}
            >
              <Wishlist />
            </Flex>
          </PageSection>
        </SidebarContent>
      </Sidebar>
    </Page>
  );
}
