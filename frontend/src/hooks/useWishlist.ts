import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  fetchWishlist,
  addToWishlist,
  removeFromWishlist,
} from '../services/wishlist';

export const useWishlist = (userId?: string) => {
  return useQuery({
    queryKey: ['wishlist', userId],
    queryFn: () => fetchWishlist(userId),
    staleTime: 2 * 60 * 1000, // Override: wishlist changes more frequently
  });
};

export const useAddToWishlist = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      userId,
      productId,
    }: {
      userId: string;
      productId: string;
    }) => addToWishlist(userId, productId),
    onSuccess: (_, { userId }) => {
      queryClient.invalidateQueries({ queryKey: ['wishlist', userId] });
    },
  });
};

export const useRemoveFromWishlist = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      userId,
      productId,
    }: {
      userId: string;
      productId: string;
    }) => removeFromWishlist(userId, productId),
    onSuccess: (_, { userId }) => {
      queryClient.invalidateQueries({ queryKey: ['wishlist', userId] });
    },
  });
};
