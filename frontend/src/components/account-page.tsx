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
import { useAuth } from '../contexts/AuthProvider';

export function AccountPage() {
  const { user, isLoading, isAuthenticated } = useAuth();

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
    </>
  );
}
