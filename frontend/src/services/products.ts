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
    `/products/search?query=${encodeURIComponent(query)}`,
    'searchProducts'
  );
};

export const searchProductsByText = async (
  query: string,
  k: number = 5
): Promise<ProductData[]> => {
  ServiceLogger.logServiceCall('searchProductsByText', { query, k });
  return apiRequest<ProductData[]>(
    `/products/search?query=${encodeURIComponent(query)}&k=${k}`,
    'searchProductsByText'
  );
};

export const searchProductsByImageLink = async (
  imageLink: string,
  k: number = 5
): Promise<ProductData[]> => {
  ServiceLogger.logServiceCall('searchProductsByImageLink', { imageLink, k });
  return apiRequest<ProductData[]>(
    `/products/search/image_link?image_link=${encodeURIComponent(imageLink)}&k=${k}`,
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
    '/products/search/image',
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

  try {
    // Make GET request to fetch single product
    const product = await apiRequest<ProductData>(
      `/products/${productId}`,
      'fetchProduct'
    );

    return product;
  } catch (error) {
    // If API call fails, return fallback single product data
    console.warn('Product fetch failed, using fallback data:', error);
    return {
      item_id: '1',
      product_name: 'Page Not Found',
      actual_price: 0.0,
      rating: 0.0,
      category: 'No Category',
      about_product: 'Could not find product, please try again later.',
    };
  }
};
