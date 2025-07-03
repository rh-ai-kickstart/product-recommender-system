import {
  Flex,
  FlexItem,
  MastheadBrand,
  MastheadContent,
  MastheadMain,
  MastheadToggle,
  Nav,
  NavItem,
  NavList,
  Masthead as PFMasthead,
  PageToggleButton,
  SearchInput,
  Title,
  Toolbar,
  ToolbarContent,
  ToolbarGroup,
  ToolbarItem,
} from "@patternfly/react-core";

import { Link, useLocation } from "@tanstack/react-router";
import { BarsIcon, CogIcon } from "@patternfly/react-icons";

export const themeStorageKey = "app-theme";

interface MastheadProps {
  showSidebarToggle?: boolean;
  isSidebarOpen?: boolean;
  onSidebarToggle?: () => void;
}

export function Masthead({
  showSidebarToggle = false,
  isSidebarOpen = false,
  onSidebarToggle,
}: MastheadProps) {
  const location = useLocation();

  const nav = (
    <Nav variant="horizontal" aria-label="Main Nav">
      <NavList>
        <NavItem itemId={0} isActive={location.pathname == "/"} to="#">
          <Link to="/">
            <Flex
              direction={{ default: "row" }}
              alignItems={{ default: "alignItemsCenter" }}
              gap={{ default: "gapSm" }}
            >
              <FlexItem>Home</FlexItem>
            </Flex>
          </Link>
        </NavItem>
        <NavItem
          icon={<CogIcon />}
          itemId={1}
          isActive={location.pathname.startsWith("/account")}
          to="#"
        >
          <Link to="/account">
            <Flex
              direction={{ default: "row" }}
              alignItems={{ default: "alignItemsCenter" }}
              gap={{ default: "gapSm" }}
            >
              <FlexItem>My Account</FlexItem>
            </Flex>
          </Link>
        </NavItem>
      </NavList>
    </Nav>
  );

  const toggle =
    showSidebarToggle && onSidebarToggle ? (
      <PageToggleButton
        variant="plain"
        aria-label="Global navigation"
        isSidebarOpen={isSidebarOpen}
        onSidebarToggle={onSidebarToggle}
        id="main-padding-nav-toggle"
      >
        <BarsIcon />
      </PageToggleButton>
    ) : null;

  const toolbar = (
    <Toolbar
      inset={{
        default: "insetSm",
        md: "insetMd",
        lg: "insetLg",
        xl: "insetXl",
        "2xl": "inset2xl",
      }}
      isFullHeight
    >
      <ToolbarContent>
        <ToolbarGroup align={{ default: "alignEnd" }}>
          <SearchInput aria-label="Sticky example search input" />
          <ToolbarItem>{nav}</ToolbarItem>
        </ToolbarGroup>
      </ToolbarContent>
    </Toolbar>
  );

  const title =
    location.pathname == "/search"
      ? "Search"
      : location.pathname == "/account"
        ? "My Account"
        : location.pathname == "/login"
          ? "Login"
          : location.pathname == "/signup"
            ? "Sign Up"
            : "Product Recommendations"

  return (
    <PFMasthead>
      <MastheadMain>
        {showSidebarToggle && <MastheadToggle>{toggle}</MastheadToggle>}
        <MastheadBrand data-codemods>
          <Title headingLevel="h4">{title}</Title>
        </MastheadBrand>
      </MastheadMain>
      <MastheadContent>{toolbar}</MastheadContent>
    </PFMasthead>
  );
}
