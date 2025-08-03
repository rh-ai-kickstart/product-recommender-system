import { apiRequest, ServiceLogger } from './api';

export interface Feedback {
  userId: string;
  productId: string;
  rating: number;
  title?: string;
  comment?: string;
}

export const submitFeedback = async (feedback: Feedback): Promise<void> => {
  ServiceLogger.logServiceCall('submitFeedback', { feedback });
  return apiRequest<void>('/feedback', 'submitFeedback', {
    method: 'POST',
    body: feedback,
  });
};
