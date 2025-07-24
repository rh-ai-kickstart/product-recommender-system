import {
  PageSection,
  Title,
  Card,
  CardBody,
  DescriptionList,
  DescriptionListGroup,
  DescriptionListTerm,
  DescriptionListDescription,
  Spinner,
  Alert,
} from '@patternfly/react-core';
import { Carousel } from './Carousel/Carousel';
import { GalleryView } from './Gallery';
import { useAuth } from '../contexts/AuthProvider';

export function AccountPage() {
  const { user, isLoading, isAuthenticated } = useAuth();

  // Placeholder empty arrays for demo
  const productsRecommended: any[] = [];
  const highlyRecProducts: any[] = [];
  const trendingProducts: any[] = [];

  if (isLoading) {
    return (
      <PageSection>
        <Spinner size='lg' />
      </PageSection>
    );
  }

  if (!isAuthenticated || !user) {
    return (
      <PageSection>
        <Alert variant='warning' title='Authentication Required'>
          Please log in to view your account page.
        </Alert>
      </PageSection>
    );
  }

  return (
    <>
      {/* User Profile Section */}
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={'h1'} style={{ marginTop: '15px' }}>
          My Account
        </Title>
        <Card style={{ marginTop: '20px' }}>
          <CardBody>
            <Title headingLevel='h2' size='lg'>
              Profile Information
            </Title>
            <DescriptionList isHorizontal>
              <DescriptionListGroup>
                <DescriptionListTerm>Email</DescriptionListTerm>
                <DescriptionListDescription>
                  {user.email}
                </DescriptionListDescription>
              </DescriptionListGroup>
              <DescriptionListGroup>
                <DescriptionListTerm>User ID</DescriptionListTerm>
                <DescriptionListDescription>
                  {user.user_id}
                </DescriptionListDescription>
              </DescriptionListGroup>
              <DescriptionListGroup>
                <DescriptionListTerm>Age</DescriptionListTerm>
                <DescriptionListDescription>
                  {user.age}
                </DescriptionListDescription>
              </DescriptionListGroup>
              <DescriptionListGroup>
                <DescriptionListTerm>Gender</DescriptionListTerm>
                <DescriptionListDescription>
                  {user.gender}
                </DescriptionListDescription>
              </DescriptionListGroup>
              <DescriptionListGroup>
                <DescriptionListTerm>Member Since</DescriptionListTerm>
                <DescriptionListDescription>
                  {new Date(user.signup_date).toLocaleDateString()}
                </DescriptionListDescription>
              </DescriptionListGroup>
              <DescriptionListGroup>
                <DescriptionListTerm>Preferences</DescriptionListTerm>
                <DescriptionListDescription>
                  {user.preferences || 'No preferences set'}
                </DescriptionListDescription>
              </DescriptionListGroup>
            </DescriptionList>
          </CardBody>
        </Card>
      </PageSection>

      {/* Product Recommendations Section */}
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={'h2'} style={{ marginTop: '15px' }}>
          Product Recommendations for {user.email.split('@')[0]}
        </Title>
      </PageSection>
      <Carousel products={productsRecommended} />

      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={'h2'} style={{ marginTop: '15px' }}>
          Trending Products
        </Title>
        <GalleryView products={trendingProducts} />
      </PageSection>

      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={'h2'} style={{ marginTop: '15px' }}>
          Highly Recommended
        </Title>
        <Carousel products={highlyRecProducts} />
      </PageSection>
    </>
  );
}
