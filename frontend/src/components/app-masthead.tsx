import {
  MastheadContent,
  Masthead as PFMasthead,
  Title,
  Toolbar,
  ToolbarContent,
  ToolbarGroup,
  ToolbarItem,
} from '@patternfly/react-core';

import { Link, useLocation } from '@tanstack/react-router';
import { Search } from './search';
import { UserDropdown } from './user-dropdown';
import { CartIcon } from './cart-icon';

export function AppMasthead() {
  const location = useLocation();
  const isPreferencesPage = location.pathname === '/preferences';

  const toolbar = (
    <Toolbar isFullHeight>
      <ToolbarContent>
        <ToolbarGroup>
          <ToolbarItem>
            <Link to='/' style={{ textDecoration: 'none', color: 'inherit' }}>
              <Title headingLevel='h4'>Product Recommendations</Title>
            </Link>
          </ToolbarItem>
        </ToolbarGroup>
        {!isPreferencesPage && (
          <ToolbarGroup
            className='pf-v6-u-w-100 pf-v6-u-w-75-on-md pf-v6-u-px-xl-on-md'
            variant='filter-group'
            align={{ default: 'alignCenter' }}
          >
            <ToolbarItem className='pf-v6-u-w-100'>
              <Search />
            </ToolbarItem>
          </ToolbarGroup>
        )}
        <ToolbarGroup
          variant='action-group'
          className='pf-v6-u-display-none pf-v6-u-display-block-on-md'
        >
          <ToolbarItem>
            <CartIcon />
          </ToolbarItem>
          <ToolbarItem>
            <UserDropdown />
          </ToolbarItem>
        </ToolbarGroup>
      </ToolbarContent>
    </Toolbar>
  );

  return (
    <PFMasthead>
      <MastheadContent>{toolbar}</MastheadContent>
    </PFMasthead>
  );
}
