import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  fetchCart,
  addToCart,
  updateCart,
  removeFromCart,
  editCart,
} from '../services/cart';

export const useCart = (userId: string) => {
  return useQuery({
    queryKey: ['cart', userId],
    queryFn: () => fetchCart(userId),
    enabled: !!userId,
    staleTime: 30 * 1000, // Override: cart changes very frequently
  });
};

export const useAddToCart = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: addToCart,
    onSuccess: (_, cartItem) => {
      queryClient.invalidateQueries({ queryKey: ['cart', cartItem.user_id] });
    },
  });
};

export const useUpdateCart = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: updateCart,
    onSuccess: (_, cartItem) => {
      queryClient.invalidateQueries({ queryKey: ['cart', cartItem.user_id] });
    },
  });
};

export const useRemoveFromCart = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: removeFromCart,
    onSuccess: (_, cartItem) => {
      queryClient.invalidateQueries({ queryKey: ['cart', cartItem.user_id] });
    },
  });
};

// Legacy editCart hook for backward compatibility
export const useEditCart = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: editCart,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cart'] });
    },
  });
};
