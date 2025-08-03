import { apiRequest, ServiceLogger } from './api';

export interface OrderItem {
  product_id: string;
  quantity: number;
  price?: number;
}

export interface CheckoutRequest {
  user_id: string;
  items: OrderItem[];
}

export interface Order {
  order_id: number;
  user_id: string;
  items: OrderItem[];
  total_amount: number;
  order_date: string;
  status: string;
}

export const checkout = async (request: CheckoutRequest): Promise<Order> => {
  ServiceLogger.logServiceCall('checkout', { request });
  return apiRequest<Order>('/checkout', 'checkout', {
    method: 'POST',
    body: request,
  });
};

export const fetchOrderHistory = async (userId: string): Promise<Order[]> => {
  ServiceLogger.logServiceCall('fetchOrderHistory', { userId });
  return apiRequest<Order[]>(`/orders/${userId}`, 'fetchOrderHistory');
};
