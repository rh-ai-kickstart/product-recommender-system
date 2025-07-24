import type { ProductData } from '../types';
import { apiRequest, ServiceLogger } from './api';

/**
 * Fetch personalized recommendations for users with existing interaction history
 * These recommendations use the user's past behavior to suggest relevant products
 */
export const fetchExistingUserRecommendations = async (
  userId: string
): Promise<ProductData[]> => {
  ServiceLogger.logServiceCall('fetchExistingUserRecommendations', { userId });
  return apiRequest<ProductData[]>(
    `/api/recommendations/${userId}`,
    'fetchExistingUserRecommendations'
  );
};

/**
 * Fetch recommendations for new users without interaction history (cold start problem)
 * These recommendations are based on general popularity or category preferences
 */
export const fetchNewUserRecommendations = async (
  numRecommendations: number = 10
): Promise<ProductData[]> => {
  ServiceLogger.logServiceCall('fetchNewUserRecommendations', {
    numRecommendations,
  });
  return apiRequest<ProductData[]>(
    '/api/recommendations',
    'fetchNewUserRecommendations',
    {
      method: 'POST',
      body: JSON.stringify({ num_recommendations: numRecommendations }),
    }
  );
};
