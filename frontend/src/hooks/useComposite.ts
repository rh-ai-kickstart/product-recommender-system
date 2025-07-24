// Composite hooks that combine multiple operations for better DX
import { useAuth } from '../contexts/AuthProvider';
import { useProduct, useProductSearch } from './useProducts';
import {
  usePersonalizedRecommendations
} from './useRecommendations';
import type { ProductData } from '../types';
import { useCart, useAddToCart } from './useCart';
import { useWishlist, useAddToWishlist, useRemoveFromWishlist } from './useWishlist';
import { useRecordProductClick } from './useInteractions';

/**
 * Hook that provides all product-related actions for a specific product
 * Combines product data, cart operations, wishlist operations, and interaction tracking
 * Requires an authenticated user for cart/wishlist operations
 */
export const useProductActions = (productId: string) => {
  const { user } = useAuth();
  const userId = user?.user_id || '';

  // Data hooks
  const productQuery = useProduct(productId);
  const cartQuery = useCart(userId);
  const wishlistQuery = useWishlist(userId);

  // Mutation hooks
  const addToCartMutation = useAddToCart();
  const addToWishlistMutation = useAddToWishlist();
  const removeFromWishlistMutation = useRemoveFromWishlist();
  const recordClickMutation = useRecordProductClick();

  // Derived state
  const isInCart = cartQuery.data?.some((item) => item.product_id === productId) ?? false;
  const isInWishlist =
    wishlistQuery.data?.some((product) => product.id.toString() === productId) ?? false;

  // Composite actions
  const addToCart = (quantity: number = 1) => {
    if (!userId) throw new Error('User must be authenticated to add to cart');

    return addToCartMutation.mutate({
      user_id: userId,
      product_id: productId,
      quantity,
    });
  };

  const toggleWishlist = () => {
    if (!userId) throw new Error('User must be authenticated to modify wishlist');

    if (isInWishlist) {
      return removeFromWishlistMutation.mutate({ userId, productId });
    } else {
      return addToWishlistMutation.mutate({ userId, productId });
    }
  };

  const recordClick = () => {
    recordClickMutation.mutate(productId);
  };

  return {
    // Data
    product: productQuery.data,
    isLoading: productQuery.isLoading,
    error: productQuery.error,

    // State
    isInCart,
    isInWishlist,

    // Actions
    addToCart,
    toggleWishlist,
    recordClick,

    // Loading states
    isAddingToCart: addToCartMutation.isPending,
    isTogglingWishlist: addToWishlistMutation.isPending || removeFromWishlistMutation.isPending,

    // Raw mutations for advanced use
    mutations: {
      addToCart: addToCartMutation,
      addToWishlist: addToWishlistMutation,
      removeFromWishlist: removeFromWishlistMutation,
      recordClick: recordClickMutation,
    },
  };
};

/**
 * Lightweight hook for product cards in lists (homepage, search results, catalog)
 * Provides cart/wishlist actions without fetching product data (assumes you already have it)
 * Optimized for use in multiple product cards simultaneously
 */
export const useProductCardActions = (productId: string) => {
  const { user } = useAuth();
  const userId = user?.user_id || '';

  // Only fetch user's cart and wishlist data (shared across all cards)
  const cartQuery = useCart(userId);
  const wishlistQuery = useWishlist(userId);

  // Mutation hooks (can be used by multiple cards simultaneously)
  const addToCartMutation = useAddToCart();
  const addToWishlistMutation = useAddToWishlist();
  const removeFromWishlistMutation = useRemoveFromWishlist();
  const recordClickMutation = useRecordProductClick();

  // Derived state for this specific product
  const isInCart = cartQuery.data?.some((item) => item.product_id === productId) ?? false;
  const isInWishlist =
    wishlistQuery.data?.some((product) => product.id.toString() === productId) ?? false;

  // Lightweight actions (no product data fetching)
  const addToCart = (quantity: number = 1) => {
    if (!userId) throw new Error('User must be authenticated to add to cart');

    return addToCartMutation.mutate({
      user_id: userId,
      product_id: productId,
      quantity,
    });
  };

  const toggleWishlist = () => {
    if (!userId) throw new Error('User must be authenticated to modify wishlist');

    if (isInWishlist) {
      return removeFromWishlistMutation.mutate({ userId, productId });
    } else {
      return addToWishlistMutation.mutate({ userId, productId });
    }
  };

  const recordClick = () => {
    recordClickMutation.mutate(productId);
  };

  return {
    // State (no product data - you already have it from the list)
    isInCart,
    isInWishlist,
    isAuthenticated: !!userId,

    // Actions
    addToCart,
    toggleWishlist,
    recordClick,

    // Loading states
    isAddingToCart: addToCartMutation.isPending,
    isTogglingWishlist: addToWishlistMutation.isPending || removeFromWishlistMutation.isPending,

    // User-specific loading states (useful for showing which card is being acted upon)
    isCartLoading: cartQuery.isLoading,
    isWishlistLoading: wishlistQuery.isLoading,
  };
};

// ============================
// LIST-LEVEL HOOKS WITH ACTIONS
// ============================

/**
 * Hook for recommendations with built-in cart/wishlist actions
 * Perfect for homepage - gets recommendations AND provides actions for each product
 */
export const useRecommendationsWithActions = () => {
  const { user } = useAuth();
  const userId = user?.user_id || '';

  // Get personalized recommendations
  const recommendationsQuery = usePersonalizedRecommendations();

  // Get user's cart/wishlist data once for the entire list
  const cartQuery = useCart(userId);
  const wishlistQuery = useWishlist(userId);

  // Shared mutation hooks
  const addToCartMutation = useAddToCart();
  const addToWishlistMutation = useAddToWishlist();
  const removeFromWishlistMutation = useRemoveFromWishlist();
  const recordClickMutation = useRecordProductClick();

  // Factory function to create actions for any product in the list
  const createProductActions = (productId: string) => {
    const isInCart = cartQuery.data?.some((item) => item.product_id === productId) ?? false;
    const isInWishlist =
      wishlistQuery.data?.some((product) => product.id.toString() === productId) ?? false;

    return {
      isInCart,
      isInWishlist,
      isAuthenticated: !!userId,
      addToCart: (quantity: number = 1) => {
        if (!userId) throw new Error('User must be authenticated to add to cart');
        return addToCartMutation.mutate({ user_id: userId, product_id: productId, quantity });
      },
      toggleWishlist: () => {
        if (!userId) throw new Error('User must be authenticated to modify wishlist');
        if (isInWishlist) {
          return removeFromWishlistMutation.mutate({ userId, productId });
        } else {
          return addToWishlistMutation.mutate({ userId, productId });
        }
      },
      recordClick: () => recordClickMutation.mutate(productId),
      isAddingToCart: addToCartMutation.isPending,
      isTogglingWishlist: addToWishlistMutation.isPending || removeFromWishlistMutation.isPending,
    };
  };

  return {
    // List data
    products: recommendationsQuery.data || [],
    isLoading: recommendationsQuery.isLoading,
    error: recommendationsQuery.error,

    // Action factory
    createProductActions,

    // Convenience: products with actions pre-attached
    productsWithActions:
      recommendationsQuery.data?.map((product: ProductData) => ({
        ...product,
        actions: createProductActions(product.id.toString()),
      })) || [],

    // Global loading states
    isCartLoading: cartQuery.isLoading,
    isWishlistLoading: wishlistQuery.isLoading,
  };
};

/**
 * Hook for search results with built-in cart/wishlist actions
 * Perfect for search pages - gets search results AND provides actions for each product
 */
export const useSearchWithActions = (query: string, enabled: boolean = true) => {
  const { user } = useAuth();
  const userId = user?.user_id || '';

  // Get the search results
  const searchQuery = useProductSearch(query, enabled);

  // Get user's cart/wishlist data once for the entire list
  const cartQuery = useCart(userId);
  const wishlistQuery = useWishlist(userId);

  // Shared mutation hooks
  const addToCartMutation = useAddToCart();
  const addToWishlistMutation = useAddToWishlist();
  const removeFromWishlistMutation = useRemoveFromWishlist();
  const recordClickMutation = useRecordProductClick();

  // Factory function to create actions for any product in the list
  const createProductActions = (productId: string) => {
    const isInCart = cartQuery.data?.some((item) => item.product_id === productId) ?? false;
    const isInWishlist =
      wishlistQuery.data?.some((product) => product.id.toString() === productId) ?? false;

    return {
      isInCart,
      isInWishlist,
      isAuthenticated: !!userId,
      addToCart: (quantity: number = 1) => {
        if (!userId) throw new Error('User must be authenticated to add to cart');
        return addToCartMutation.mutate({ user_id: userId, product_id: productId, quantity });
      },
      toggleWishlist: () => {
        if (!userId) throw new Error('User must be authenticated to modify wishlist');
        if (isInWishlist) {
          return removeFromWishlistMutation.mutate({ userId, productId });
        } else {
          return addToWishlistMutation.mutate({ userId, productId });
        }
      },
      recordClick: () => recordClickMutation.mutate(productId),
      isAddingToCart: addToCartMutation.isPending,
      isTogglingWishlist: addToWishlistMutation.isPending || removeFromWishlistMutation.isPending,
    };
  };

  return {
    // List data
    products: searchQuery.data || [],
    isLoading: searchQuery.isLoading,
    error: searchQuery.error,

    // Action factory
    createProductActions,

    // Convenience: products with actions pre-attached
    productsWithActions:
      searchQuery.data?.map((product: ProductData) => ({
        ...product,
        actions: createProductActions(product.id.toString()),
      })) || [],

    // Global loading states
    isCartLoading: cartQuery.isLoading,
    isWishlistLoading: wishlistQuery.isLoading,
  };
};
