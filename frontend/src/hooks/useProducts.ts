import { useQuery } from '@tanstack/react-query';
import {
  fetchProduct,
  searchProducts,
  searchProductsByText,
  searchProductsByImageLink,
} from '../services/products';

export const useProduct = (productId: string) => {
  return useQuery({
    queryKey: ['products', productId],
    queryFn: () => fetchProduct(productId),
    enabled: !!productId,
    staleTime: 10 * 60 * 1000, // Override: product details change less frequently
  });
};

export const useProductSearch = (query: string, enabled: boolean = true) => {
  return useQuery({
    queryKey: ['products', 'search', query],
    queryFn: () => searchProducts(query),
    enabled: enabled && !!query && query.trim().length > 0,
    staleTime: 2 * 60 * 1000, // Override: search results change more frequently
  });
};

export const useProductSearchByText = (
  query: string,
  k: number = 5,
  enabled: boolean = true
) => {
  return useQuery({
    queryKey: ['products', 'search', 'text', query, k],
    queryFn: () => searchProductsByText(query, k),
    enabled: enabled && !!query && query.trim().length > 0,
    staleTime: 2 * 60 * 1000,
  });
};

export const useProductSearchByImageLink = (
  imageLink: string,
  k: number = 5,
  enabled: boolean = true
) => {
  return useQuery({
    queryKey: ['products', 'search', 'image-link', imageLink, k],
    queryFn: () => searchProductsByImageLink(imageLink, k),
    enabled: enabled && !!imageLink,
  });
};
