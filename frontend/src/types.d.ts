export interface ProductData {
  item_id: string;
  product_name: string;
  category: string;
  about_product?: string;
  img_link?: string;
  discount_percentage?: number;
  discounted_price?: number;
  actual_price: number;
  product_link?: string;
  rating_count?: number;
  rating?: number;
}

export interface CartItem {
  userId: string;
  productId: string;
}

export interface User {
  user_id: string;
  email: string;
  age: number;
  gender: string;
  signup_date: string; // Changed from date to string to match backend
  preferences: string;
  views?: string[]; // Added optional views array
}

// Auth-related types
export interface LoginRequest {
  email: string;
  password: string;
}

export interface SignUpRequest {
  email: string;
  password: string;
  age: number;
  gender: string;
}

export interface AuthResponse {
  user: User;
  token: string;
}

export interface AuthError {
  detail: string;
}
