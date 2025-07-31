import {
  PageSection,
  TreeView,
  type TreeViewDataItem,
  Sidebar,
  SidebarPanel,
  SidebarContent,
  Flex,
} from '@patternfly/react-core';
import { createFileRoute } from '@tanstack/react-router';
import { AccountPage } from '../../components/account-page';

export const Route = createFileRoute('/_protected/account')({
  component: Account,
});

const options: TreeViewDataItem[] = [
  {
    name: 'My Account',
    id: 'account',
    children: [
      {
        name: 'Profile',
        id: 'profile',
      },
      {
        name: 'Wishlist',
        id: 'saved-products',
      },
    ],
    defaultExpanded: true,
  },
];

function Account() {
  return (
    <Sidebar>
      <SidebarPanel variant='sticky'>
        <TreeView
          aria-label='account navigation'
          data={options}
          allExpanded={true}
          hasGuides={true}
          // onSelect={onSelect}
        />
      </SidebarPanel>
      <SidebarContent>
        <PageSection>
          <Flex
            direction={{ default: 'column' }}
            spaceItems={{ default: 'spaceItemsMd' }}
          >
            <AccountPage />
          </Flex>
        </PageSection>
      </SidebarContent>
    </Sidebar>
  );
}
