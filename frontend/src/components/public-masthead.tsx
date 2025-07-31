import {
  MastheadBrand,
  MastheadMain,
  Masthead as PFMasthead,
  Title,
} from '@patternfly/react-core';

import { useLocation } from '@tanstack/react-router';

export function PublicMasthead() {
  const location = useLocation();

  const title =
    location.pathname === '/login'
      ? 'Login'
      : location.pathname === '/signup'
        ? 'Sign Up'
        : 'Product Recommendations';

  return (
    <PFMasthead>
      <MastheadMain>
        <MastheadBrand data-codemods>
          <Title headingLevel='h4'>{title}</Title>
        </MastheadBrand>
      </MastheadMain>
    </PFMasthead>
  );
}
