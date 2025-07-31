import { PageSection, Title, Spinner, Alert } from '@patternfly/react-core';
import { GalleryView } from './Gallery';
import { usePersonalizedRecommendations } from '../hooks/useRecommendations';
import { useAuth } from '../contexts/AuthProvider';

export function LandingPage() {
  const { isAuthenticated } = useAuth();
  const { data, isLoading, error } = usePersonalizedRecommendations();

  // If not authenticated, show a message prompting to log in
  if (!isAuthenticated) {
    return (
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel='h1' style={{ marginTop: '15px' }}>
          Welcome to Product Recommendations
        </Title>
        <Alert variant='info' title='Authentication Required'>
          Please log in to see personalized product recommendations tailored
          just for you!
        </Alert>
      </PageSection>
    );
  }

  if (isLoading) {
    return (
      <PageSection>
        <Spinner size='lg' />
      </PageSection>
    );
  }

  if (error) {
    return (
      <PageSection>
        <Alert variant='danger' title='Error'>
          Sorry, we couldn't load your personalized recommendations right now.
          Please try again later.
        </Alert>
      </PageSection>
    );
  }

  return (
    <>
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={'h1'} style={{ marginTop: '15px' }}>
          Recommended for You
        </Title>
      </PageSection>
      <GalleryView products={data || []} />
    </>
  );
}
