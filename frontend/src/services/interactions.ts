import { apiRequest, ServiceLogger } from './api';

export interface InteractionRequest {
  item_id: string;
  interaction_type: string;
  rating?: number;
  review_title?: string;
  review_content?: string;
  quantity?: number;
}

export interface InteractionResponse {
  message: string;
  interaction_id: string;
}

export const logInteraction = async (
  interaction: InteractionRequest
): Promise<InteractionResponse> => {
  ServiceLogger.logServiceCall('logInteraction', { interaction });
  return apiRequest<InteractionResponse>(
    '/api/interactions',
    'logInteraction',
    {
      method: 'POST',
      body: interaction,
    }
  );
};

export const recordProductClick = async (productId: string): Promise<void> => {
  ServiceLogger.logServiceCall('recordProductClick', { productId });
  return apiRequest<void>(
    `/api/products/${productId}/interactions/click`,
    'recordProductClick',
    {
      method: 'POST',
    }
  );
};
