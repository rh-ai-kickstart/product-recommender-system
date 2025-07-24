import {
  MastheadBrand,
  MastheadContent,
  MastheadMain,
  MastheadToggle,
  Masthead as PFMasthead,
  PageToggleButton,
  Title,
  Toolbar,
  ToolbarContent,
  ToolbarGroup,
  ToolbarItem,
} from '@patternfly/react-core';

import { useLocation } from '@tanstack/react-router';
import { BarsIcon } from '@patternfly/react-icons';
import { Search } from './search';
import { UserDropdown } from './user-dropdown';

interface AppMastheadProps {
  showSidebarToggle?: boolean;
  isSidebarOpen?: boolean;
  onSidebarToggle?: () => void;
}

export function AppMasthead({
  showSidebarToggle = false,
  isSidebarOpen = false,
  onSidebarToggle,
}: AppMastheadProps) {
  const location = useLocation();

  const toggle =
    showSidebarToggle && onSidebarToggle ? (
      <PageToggleButton
        variant='plain'
        aria-label='Global navigation'
        isSidebarOpen={isSidebarOpen}
        onSidebarToggle={onSidebarToggle}
        id='main-padding-nav-toggle'
      >
        <BarsIcon />
      </PageToggleButton>
    ) : null;

  const toolbar = (
    <Toolbar isFullHeight>
      <ToolbarContent>
        <ToolbarGroup
          className='pf-v6-u-w-100 pf-v6-u-w-75-on-md pf-v6-u-px-xl-on-md'
          variant='filter-group'
          align={{ default: 'alignCenter' }}
        >
          <ToolbarItem className='pf-v6-u-w-100'>
            <Search />
          </ToolbarItem>
        </ToolbarGroup>
        <ToolbarGroup
          variant='action-group'
          className='pf-v6-u-display-none pf-v6-u-display-block-on-md'
        >
          <ToolbarItem>
            <UserDropdown />
          </ToolbarItem>
        </ToolbarGroup>
      </ToolbarContent>
    </Toolbar>
  );

  const title =
    location.pathname === '/search'
      ? 'Search'
      : location.pathname === '/account'
        ? 'My Account'
        : 'Product Recommendations';

  return (
    <PFMasthead>
      <MastheadMain>
        {showSidebarToggle && <MastheadToggle>{toggle}</MastheadToggle>}
        <MastheadBrand data-codemods>
          <Title headingLevel='h4'>{title}</Title>
        </MastheadBrand>
      </MastheadMain>
      <MastheadContent>{toolbar}</MastheadContent>
    </PFMasthead>
  );
}
