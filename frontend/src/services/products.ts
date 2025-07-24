import type { ProductData } from '../types';
import { apiRequest, ServiceLogger } from './api';

export const searchProducts = async (query: string): Promise<ProductData[]> => {
  ServiceLogger.logServiceCall('searchProducts', { query });

  if (!query || query.trim().length === 0) {
    ServiceLogger.logServiceWarning(
      'searchProducts',
      'Empty search query provided'
    );
    return [];
  }

  return apiRequest<ProductData[]>(
    `/api/products/search?query=${encodeURIComponent(query)}`,
    'searchProducts'
  );
};

export const searchProductsByText = async (
  query: string,
  k: number = 5
): Promise<ProductData[]> => {
  ServiceLogger.logServiceCall('searchProductsByText', { query, k });
  return apiRequest<ProductData[]>(
    `/api/products/search?query=${encodeURIComponent(query)}&k=${k}`,
    'searchProductsByText'
  );
};

export const searchProductsByImageLink = async (
  imageLink: string,
  k: number = 5
): Promise<ProductData[]> => {
  ServiceLogger.logServiceCall('searchProductsByImageLink', { imageLink, k });
  return apiRequest<ProductData[]>(
    `/api/products/search/image_link?image_link=${encodeURIComponent(imageLink)}&k=${k}`,
    'searchProductsByImageLink'
  );
};

export const searchProductsByImage = async (
  imageFile: File,
  k: number = 5
): Promise<ProductData[]> => {
  ServiceLogger.logServiceCall('searchProductsByImage', {
    fileName: imageFile.name,
    k,
  });

  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('k', k.toString());

  return apiRequest<ProductData[]>(
    '/api/products/search/image',
    'searchProductsByImage',
    {
      method: 'POST',
      body: formData,
      headers: {}, // Let the browser set the Content-Type for FormData
    }
  );
};

export const fetchProduct = async (productId: string): Promise<ProductData> => {
  ServiceLogger.logServiceCall('fetchProduct', { productId });
  return apiRequest<ProductData>(`/api/products/${productId}`, 'fetchProduct');
};
