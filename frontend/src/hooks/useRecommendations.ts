import { useQuery } from '@tanstack/react-query';
import {
  fetchExistingUserRecommendations,
  fetchNewUserRecommendations,
} from '../services/recommendations';
import { useAuth } from '../contexts/AuthProvider';

/**
 * Smart recommendations hook that automatically chooses the right recommendation type
 * - For users with interaction history: uses personalized recommendations
 * - For new users without history: uses new user recommendations based on preferences
 * - Requires authentication (redirects to login if not authenticated)
 */
export const usePersonalizedRecommendations = () => {
  const { user, isAuthenticated } = useAuth();

  // For now, we'll treat all authenticated users as "new" until we have
  // a way to determine if they have interaction history
  // In the future, this could check user.views.length or call an endpoint
  const hasInteractionHistory = false; // TODO: Implement history check

  return useQuery({
    queryKey: ['recommendations', 'personalized', user?.user_id],
    queryFn: () => {
      if (!user?.user_id) {
        throw new Error(
          'User authentication required for personalized recommendations'
        );
      }

      if (hasInteractionHistory) {
        return fetchExistingUserRecommendations(user.user_id);
      } else {
        return fetchNewUserRecommendations(10);
      }
    },
    enabled: isAuthenticated && !!user?.user_id,
  });
};

// Recommendations for users with existing interaction history
export const useExistingUserRecommendations = (userId: string) => {
  return useQuery({
    queryKey: ['recommendations', 'existing-user', userId],
    queryFn: () => fetchExistingUserRecommendations(userId),
    enabled: !!userId,
  });
};

// Recommendations for users without interaction history (cold start)
export const useNewUserRecommendations = (numRecommendations: number = 10) => {
  return useQuery({
    queryKey: ['recommendations', 'new-user', numRecommendations],
    queryFn: () => fetchNewUserRecommendations(numRecommendations),
  });
};
