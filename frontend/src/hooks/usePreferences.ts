import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchNewPreferences, setPreferences } from '../services/preferences';
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
    mutationFn: (preferences: PreferencesRequest) => setPreferences(preferences),
    onSuccess: (authResponse) => {
      // Update preferences cache
      queryClient.setQueryData(['preferences'], authResponse.user.preferences);

      // Update current user data in auth context
      queryClient.setQueryData(['currentUser'], authResponse.user);

      // Invalidate recommendations since preferences changed
      queryClient.invalidateQueries({ queryKey: ['recommendations'] });
      // Get redirect path from URL params or default to home
      const searchParams = new URLSearchParams(window.location.search);
      const redirectPath = searchParams.get('redirect') || '/';
      navigate({ to: redirectPath });
    },
  });
};

// Re-export the type for convenience
export type { PreferencesRequest };
