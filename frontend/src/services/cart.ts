import { apiRequest, ServiceLogger } from './api';

export interface CartItem {
  user_id: string;
  product_id: string;
  quantity?: number;
}

export const fetchCart = async (userId: string): Promise<CartItem[]> => {
  ServiceLogger.logServiceCall('fetchCart', { userId });
  return apiRequest<CartItem[]>(`/cart/${userId}`, 'fetchCart');
};

export const addToCart = async (cartItem: CartItem): Promise<void> => {
  ServiceLogger.logServiceCall('addToCart', { cartItem });
  return apiRequest<void>('/cart', 'addToCart', {
    method: 'POST',
    body: cartItem,
  });
};

export const updateCart = async (cartItem: CartItem): Promise<void> => {
  ServiceLogger.logServiceCall('updateCart', { cartItem });
  return apiRequest<void>('/cart', 'updateCart', {
    method: 'PUT',
    body: cartItem,
  });
};

export const removeFromCart = async (cartItem: CartItem): Promise<void> => {
  ServiceLogger.logServiceCall('removeFromCart', { cartItem });
  return apiRequest<void>('/cart', 'removeFromCart', {
    method: 'DELETE',
    body: cartItem,
  });
};

// Keep the original editCart function for backward compatibility
export const editCart = async (cartItem: any): Promise<any> => {
  ServiceLogger.logServiceCall('editCart', { cartItem });
  return apiRequest<any>('/cart', 'editCart', {
    method: 'POST',
    body: cartItem,
  });
};
