import type { ProductData } from '../types';
import { apiRequest, ServiceLogger } from './api';

export const fetchWishlist = async (
  userId?: string
): Promise<ProductData[]> => {
  ServiceLogger.logServiceCall('fetchWishlist', { userId });

  // Use placeholder user_id for now, similar to current implementation
  const endpoint = userId
    ? `/api/wishlist/${userId}`
    : '/api/wishlist/{user_id}';

  const context = ServiceLogger.logServiceCall('fetchWishlist');

  try {
    return await apiRequest<ProductData[]>(endpoint, 'fetchWishlist');
  } catch (error) {
    ServiceLogger.logServiceError('fetchWishlist', error);
    throw error;
  }
};

export const addToWishlist = async (
  userId: string,
  productId: string
): Promise<void> => {
  ServiceLogger.logServiceCall('addToWishlist', { userId, productId });
  return apiRequest<void>('/api/wishlist', 'addToWishlist', {
    method: 'POST',
    body: { user_id: userId, product_id: productId },
  });
};

export const removeFromWishlist = async (
  userId: string,
  productId: string
): Promise<void> => {
  ServiceLogger.logServiceCall('removeFromWishlist', { userId, productId });
  return apiRequest<void>('/api/wishlist', 'removeFromWishlist', {
    method: 'DELETE',
    body: { user_id: userId, product_id: productId },
  });
};
