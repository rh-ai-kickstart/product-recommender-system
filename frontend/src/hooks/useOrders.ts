import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { checkout, fetchOrderHistory } from '../services/orders';
import type { CheckoutRequest } from '../services/orders';

export const useOrderHistory = (userId: string) => {
  return useQuery({
    queryKey: ['orders', userId],
    queryFn: () => fetchOrderHistory(userId),
    enabled: !!userId,
    staleTime: 10 * 60 * 1000, // Override: order history doesn't change often
  });
};

export const useCheckout = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CheckoutRequest) => checkout(request),
    onSuccess: (_order, request) => {
      // Invalidate order history for this user
      queryClient.invalidateQueries({ queryKey: ['orders', request.user_id] });

      // Clear cart after successful checkout
      queryClient.invalidateQueries({ queryKey: ['cart', request.user_id] });

      // You might also want to invalidate recommendations to reflect new purchase history
      queryClient.invalidateQueries({ queryKey: ['recommendations'] });
    },
  });
};
