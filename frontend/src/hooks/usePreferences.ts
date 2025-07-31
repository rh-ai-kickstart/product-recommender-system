import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { setPreferences, fetchNewPreferences } from '../services/preferences';
import { createNewUserRecommendations } from '../services/recommendations';
import type { PreferencesRequest } from '../services/preferences';
import { useNavigate } from '@tanstack/react-router';

export const usePreferences = () => {
  return useQuery<string[]>({
    queryKey: ['preferences'],
    queryFn: async () => {
      const result = await fetchNewPreferences();
      return result.split('|'); // Convert pipe-separated string to array
    },
    staleTime: 10 * 60 * 1000,
  });
};

export const useSetPreferences = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: (preferences: PreferencesRequest) =>
      setPreferences(preferences),
    onSuccess: async authResponse => {
      // Update user data in cache with new preferences
      queryClient.setQueryData(['currentUser'], authResponse.user);

      // Create new user recommendations via ML model
      try {
        console.log('Triggering new user recommendation generation...');
        await createNewUserRecommendations(10);
        console.log('New user recommendations created successfully');

        // Invalidate recommendations cache to force refresh
        queryClient.invalidateQueries({
          queryKey: ['recommendations', 'personalized'],
        });
        queryClient.invalidateQueries({
          queryKey: ['recommendations', 'new-user'],
        });
      } catch (error) {
        console.warn(
          'Failed to create initial recommendations via ML model:',
          error
        );
        console.log('Will fall back to existing user recommendations endpoint');
      }

      // Get redirect path from URL params or default to home
      const searchParams = new URLSearchParams(window.location.search);
      const redirectPath = searchParams.get('redirect') || '/';
      navigate({ to: redirectPath });
    },
  });
};

// Re-export the type for convenience
export type { PreferencesRequest };
