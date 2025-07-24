import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getPreferences, setPreferences } from '../services/preferences';
import type { PreferencesRequest } from '../services/preferences';

export const usePreferences = () => {
  return useQuery({
    queryKey: ['preferences'],
    queryFn: getPreferences,
    staleTime: 10 * 60 * 1000, // Override: preferences don't change often
  });
};

export const useSetPreferences = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (preferences: PreferencesRequest) =>
      setPreferences(preferences),
    onSuccess: authResponse => {
      // Update preferences cache
      queryClient.setQueryData(['preferences'], authResponse.user.preferences);

      // Update current user data in auth context
      queryClient.setQueryData(['currentUser'], authResponse.user);

      // Invalidate recommendations since preferences changed
      queryClient.invalidateQueries({ queryKey: ['recommendations'] });
    },
  });
};

// Re-export the type for convenience
export type { PreferencesRequest };
