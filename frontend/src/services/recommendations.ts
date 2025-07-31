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
 * Create new user recommendations via ML model (POST endpoint)
 * This triggers the backend to generate initial recommendations for new users
 */
export const createNewUserRecommendations = async (
  numRecommendations: number = 10
): Promise<ProductData[]> => {
  ServiceLogger.logServiceCall('createNewUserRecommendations', {
    numRecommendations,
  });

  return apiRequest<ProductData[]>(
    '/api/recommendations',
    'createNewUserRecommendations',
    {
      method: 'POST',
      body: { num_recommendations: numRecommendations },
    }
  );
};

/**
 * Fetch recommendations for new users without interaction history (cold start problem)
 * Since no ML model is running, this uses the same endpoint as existing users
 * The backend will handle the lack of interaction data gracefully
 */
export const fetchNewUserRecommendations = async (
  userId: string,
  numRecommendations: number = 10
): Promise<ProductData[]> => {
  ServiceLogger.logServiceCall('fetchNewUserRecommendations', {
    userId,
    numRecommendations,
  });

  try {
    // Use the same endpoint as existing users until backend creates model generated new user recommendations - backend will handle gracefully
    return await apiRequest<ProductData[]>(
      `/api/recommendations/${userId}`,
      'fetchNewUserRecommendations'
    );
  } catch (error) {
    // If backend fails, return mock data as fallback
    console.warn('Recommendations failed, using fallback data:', error);
    return [
      {
        item_id: '1',
        product_name: 'Sample Product 1',
        actual_price: 29.99,
        rating: 4.5,
        category: 'Sample Category',
        about_product:
          'This is a sample product while we set up your personalized recommendations.',
        img_link: 'https://via.placeholder.com/300x300?text=Product+1',
      },
      {
        item_id: '2',
        product_name: 'Sample Product 2',
        actual_price: 49.99,
        rating: 4.0,
        category: 'Sample Category',
        about_product: 'Another sample product for testing purposes.',
        img_link:
          'https://repo-avatars.githubusercontent.com/300x300?text=Product+2',
      },
    ];
  }
};
