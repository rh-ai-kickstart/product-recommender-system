import { useMutation, useQueryClient } from '@tanstack/react-query';
import { logInteraction, recordProductClick } from '../services/interactions';
import type { InteractionRequest } from '../services/interactions';

export const useLogInteraction = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (interaction: InteractionRequest) =>
      logInteraction(interaction),
    onSuccess: (_, interaction) => {
      // Interactions might affect recommendations
      queryClient.invalidateQueries({ queryKey: ['recommendations'] });

      // If it's a rating interaction, update product cache
      if (interaction.rating) {
        queryClient.invalidateQueries({
          queryKey: ['products', interaction.item_id],
        });
      }
    },
  });
};

export const useRecordProductClick = () => {
  return useMutation({
    mutationFn: (productId: string) => recordProductClick(productId),
    // Product clicks are fire-and-forget, no need to invalidate caches
  });
};

// Re-export types for convenience
export type { InteractionRequest };
