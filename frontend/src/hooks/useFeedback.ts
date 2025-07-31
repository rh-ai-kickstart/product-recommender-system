import { useMutation, useQueryClient } from '@tanstack/react-query';
import { submitFeedback } from '../services/feedback';
import type { Feedback } from '../services/feedback';

export const useSubmitFeedback = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (feedback: Feedback) => submitFeedback(feedback),
    onSuccess: (_, feedback) => {
      // Invalidate product data since rating may have changed
      queryClient.invalidateQueries({
        queryKey: ['products', feedback.productId],
      });

      // Invalidate product search results that might show this product
      queryClient.invalidateQueries({ queryKey: ['products', 'search'] });

      // Invalidate recommendations since user preferences may have changed
      queryClient.invalidateQueries({ queryKey: ['recommendations'] });
    },
  });
};

// Re-export the Feedback type for convenience
export type { Feedback };
