import type { ProductData } from "../types";

export const fetchRecommendations = async (): Promise<ProductData[]> => {
  const response = await fetch("/api/recommendations");
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  const data: unknown = await response.json();
  return data as ProductData[];
};

export const fetchWishlist= async (): Promise<ProductData[]> => {
  // sleep 3000ms
  await new Promise(resolve => setTimeout(resolve, 3000));
  const response = await fetch(`/api/wishlist/{user_id}`);
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  const data: unknown = await response.json();
  return data as ProductData[];
};

export const fetchSearch = async (): Promise<ProductData[]> => {
  const response = await fetch("/api/products/search");
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  const data: unknown = await response.json();
  return data as ProductData[];
};
