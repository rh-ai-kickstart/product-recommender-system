import type { ProductData } from '../types';

export const fetchRecommendations = async (): Promise<ProductData[]> => {
  const response = await fetch('/recommendations');
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  const data: unknown = await response.json();
  return data as ProductData[];
};

export const fetchWishlist = async (): Promise<ProductData[]> => {
  const user_id = 'user1';
  const response = await fetch(`/wishlist/${user_id}`);
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  const data: unknown = await response.json();
  return data as ProductData[];
};

export const fetchSearch = async (): Promise<ProductData[]> => {
  const response = await fetch('/products/search');
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  const data: unknown = await response.json();
  return data as ProductData[];
};

export const fetchNewPreferences = async (): Promise<string[]> => {
  const response = await fetch('/users/preferences');
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  const data: string = await response.json();
  if (!data) return [];
  return data.split('|');
};

export const addPreferences = async (
  newPreferences: string
): Promise<string> => {
  const response = await fetch('/users/preferences', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(newPreferences),
  });
  if (!response.ok) {
    console.log(newPreferences);
    throw new Error('Network response was not ok');
  }
  const data: unknown = await response.json();
  return data as string;
};
