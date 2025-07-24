# Custom Hooks Documentation

This directory contains custom React hooks that encapsulate data fetching, mutations, and complex state logic using TanStack Query.

## Directory Structure

```
hooks/
├── useAuth.ts          # Authentication hooks
├── useCart.ts          # Shopping cart operations
├── useComposite.ts     # Complex composite hooks
├── useFeedback.ts      # User feedback submission
├── useInteractions.ts  # User interaction logging
├── useOptimistic.ts    # Optimistic UI updates
├── useOrders.ts        # Order management
├── usePreferences.ts   # User preference management
├── useProducts.ts      # Product search and details
├── useRecommendations.ts # PERSONALIZED product recommendations
└── useWishlist.ts      # Wishlist management
```

## 🔐 Authentication Hooks (`useAuth.ts`)

### Overview

Dedicated hooks for authentication operations with automatic navigation and error handling.

### Available Hooks

#### `useLogin()`

```typescript
import { useLogin } from '../hooks/useAuth';

const LoginComponent = () => {
  const loginMutation = useLogin();

  const handleSubmit = async (credentials) => {
    try {
      await loginMutation.mutateAsync(credentials);
      // Automatically redirects to original page or home
    } catch (error) {
      setError(error.message);
    }
  };

  return (
    <Button
      onClick={handleSubmit}
      isLoading={loginMutation.isPending}
      isDisabled={loginMutation.isPending}
    >
      {loginMutation.isPending ? 'Logging in...' : 'Login'}
    </Button>
  );
};
```

#### `useSignup()`

```typescript
import { useSignup } from '../hooks/useAuth';

const SignupComponent = () => {
  const signupMutation = useSignup();

  const handleSubmit = async (userData) => {
    try {
      await signupMutation.mutateAsync({
        email: userData.email,
        password: userData.password,
        age: parseInt(userData.age),
        gender: userData.gender,
      });
      // Automatically redirects to home page
    } catch (error) {
      setError(error.message);
    }
  };

  return (
    <Button
      onClick={handleSubmit}
      isLoading={signupMutation.isPending}
      isDisabled={signupMutation.isPending}
    >
      {signupMutation.isPending ? 'Creating Account...' : 'Sign Up'}
    </Button>
  );
};
```

#### `useLogout()`

```typescript
import { useLogout } from '../hooks/useAuth';

const UserDropdown = () => {
  const logoutMutation = useLogout();

  const handleLogout = () => {
    logoutMutation.mutate(); // Automatically clears cache and redirects
  };

  return (
    <DropdownItem
      onClick={handleLogout}
      isDisabled={logoutMutation.isPending}
    >
      {logoutMutation.isPending ? 'Logging out...' : 'Logout'}
    </DropdownItem>
  );
};
```

### Authentication Benefits

- ✅ **Automatic redirect handling** - respects `?redirect` parameter
- ✅ **Loading states** - built-in `isPending` for UI feedback
- ✅ **Error handling** - proper error propagation
- ✅ **Cache management** - automatic cleanup on logout
- ✅ **Type safety** - full TypeScript support

### Context vs Hooks

```typescript
// ❌ Don't use context directly in components
import { useAuth } from '../contexts/AuthProvider';

// ✅ Use dedicated auth hooks instead
import { useAuth, useLogin, useLogout } from '../hooks/useAuth';

const MyComponent = () => {
  // ✅ Get user state from context
  const { user, isAuthenticated } = useAuth();

  // ✅ Use dedicated hooks for actions
  const loginMutation = useLogin();
  const logoutMutation = useLogout();
};
```

## 🎯 Personalized Recommendations (`useRecommendations.ts`)

### Philosophy: **Personalization-First**

This app is designed around personalized experiences. All recommendations are tailored to the individual user based on their preferences and interaction history.

### User Flow

1. **Sign up** → User creates account
2. **Preferences selection** → User picks interests/categories (future feature)
3. **Personalized homepage** → See recommendations based on preferences
4. **Interaction learning** → Algorithm improves based on user behavior

### Available Hooks

#### `usePersonalizedRecommendations()` - **Primary Hook**

**Use for**: Homepage, catalog, any "browse products" experience

```typescript
import { usePersonalizedRecommendations } from '../hooks/useRecommendations';

const HomePage = () => {
  const { data: products, isLoading, error } = usePersonalizedRecommendations();

  if (!isAuthenticated) {
    return <div>Please log in to see your personalized recommendations!</div>;
  }

  if (isLoading) return <ProductSkeleton />;
  if (error) return <ErrorMessage />;

  return (
    <div>
      <h1>Recommended for You</h1>
      <ProductGrid products={products} />
    </div>
  );
};
```

**Smart Logic:**

- ✅ **New users**: Gets preference-based recommendations
- ✅ **Existing users**: Gets history-based recommendations
- ✅ **Auto-selection**: Chooses the right algorithm automatically
- ✅ **Auth required**: Only works for logged-in users

#### `useExistingUserRecommendations(userId)` - **Advanced Users**

**Use for**: Users with rich interaction history

```typescript
const { data } = useExistingUserRecommendations(user.user_id);
// Returns ML-powered recommendations based on past behavior
```

#### `useNewUserRecommendations(numRecommendations)` - **New Users**

**Use for**: Users without interaction history (cold start)

```typescript
const { data } = useNewUserRecommendations(10);
// Returns preference-based recommendations for new users
```

### No Anonymous Browsing

Unlike traditional e-commerce sites, this app **requires authentication** for the core browsing experience. This enables:

- ✅ **Better recommendations** from day one
- ✅ **Preference-driven discovery** instead of generic popular items
- ✅ **Personalized learning** that improves over time
- ✅ **Seamless cart/wishlist** integration

## 🎯 Product Action Hooks

### Two Patterns for Different Use Cases

#### `useProductActions` - Detail Pages

**Use for**: Product detail pages where you need full control

```typescript
import { useProductActions } from '../hooks/useComposite';

const ProductDetailPage = ({ productId }) => {
  const {
    // State
    isInCart, isInWishlist, isAuthenticated,
    // Actions
    addToCart, toggleWishlist, recordClick,
    // Loading states
    isAddingToCart, isTogglingWishlist
  } = useProductActions(productId);

  return (
    <div>
      <Button
        onClick={() => addToCart(1)}
        isLoading={isAddingToCart}
      >
        {isInCart ? 'In Cart' : 'Add to Cart'}
      </Button>

      <Button
        onClick={toggleWishlist}
        isLoading={isTogglingWishlist}
      >
        {isInWishlist ? 'Remove from Wishlist' : 'Add to Wishlist'}
      </Button>
    </div>
  );
};
```

#### `useProductCardActions` - List Items

**Use for**: Product cards in lists (homepage, search, recommendations)

```typescript
import { useProductCardActions } from '../hooks/useComposite';

const ProductCard = ({ product }) => {
  const actions = useProductCardActions(product.id.toString());

  return (
    <div className="product-card">
      <h3 onClick={actions.recordClick}>{product.name}</h3>

      <Button
        onClick={() => actions.addToCart(1)}
        isLoading={actions.isAddingToCart}
        size="sm"
      >
        Add to Cart
      </Button>

      <IconButton
        onClick={actions.toggleWishlist}
        isLoading={actions.isTogglingWishlist}
        icon={actions.isInWishlist ? FilledHeartIcon : HeartIcon}
      />
    </div>
  );
};
```

### Performance Benefits

`useProductCardActions` is optimized for lists:

- **Shared queries**: Cart/wishlist data fetched once for entire list
- **Lightweight**: Only essential actions for card interactions
- **Efficient**: Minimal re-renders when cart/wishlist updates

## 🔗 List-Level Hooks

### Complete List Solutions

Get data + actions in one hook for entire lists:

#### `useRecommendationsWithActions`

```typescript
import { useRecommendationsWithActions } from '../hooks/useComposite';

const HomePage = () => {
  const {
    // Data
    products, isLoading, error,
    // Factory function
    createProductActions,
    // Pre-built products with actions
    productsWithActions,
  } = useRecommendationsWithActions();

  return (
    <div className="products-grid">
      {productsWithActions.map(product => (
        <ProductCard
          key={product.id}
          product={product}
          actions={product.actions} // Pre-attached actions!
        />
      ))}
    </div>
  );
};
```

#### `useSearchWithActions`

```typescript
const SearchResults = ({ query }) => {
  const { productsWithActions, isLoading } = useSearchWithActions(query);

  // Each product has pre-attached cart/wishlist actions
  return <ProductGrid products={productsWithActions} />;
};
```

## 🎯 Global Query Configuration

All hooks benefit from global defaults set in `QueryClientProvider`:

```typescript
// In __root.tsx
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
      retry: intelligentRetryLogic, // Don't retry 4xx errors
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1, // Only retry once
      networkMode: 'online',
    },
  },
});
```

**Benefits:**

- No need to specify `staleTime`, `gcTime`, `retry` in individual hooks
- Consistent behavior across the app
- Easy to modify globally

## 🚀 Best Practices

### 1. Import from Hooks Index

```typescript
// ✅ Good - uses barrel export
import { useProduct, useAddToCart, useLogin } from '../hooks';

// ❌ Avoid - direct imports
import { useProduct } from '../hooks/useProducts';
import { useAddToCart } from '../hooks/useCart';
```

### 2. Use Appropriate Hook for Context

```typescript
// ✅ Personalized browsing (primary pattern)
const { products, isLoading } = usePersonalizedRecommendations();

// ✅ Product detail pages
const { addToCart, isAddingToCart } = useProductActions(productId);

// ✅ Product cards in lists
const actions = useProductCardActions(productId);

// ✅ Complete list with actions
const { productsWithActions } = useRecommendationsWithActions();
```

### 3. Handle Loading and Error States

```typescript
const ProductList = () => {
  const { products, isLoading, error } = usePersonalizedRecommendations();

  if (isLoading) return <ProductSkeleton />;
  if (error) return <ErrorMessage error={error} />;

  return <ProductGrid products={products} />;
};
```

### 4. Authentication-First Design

```typescript
const HomePage = () => {
  const { isAuthenticated } = useAuth();
  const { products, isLoading } = usePersonalizedRecommendations();

  if (!isAuthenticated) {
    return <div>Please log in to see your personalized recommendations!</div>;
  }

  return <ProductGrid products={products} />;
};
```

This architecture provides a clean separation between data fetching, state management, and user interactions while maintaining excellent performance and a personalization-first user experience.
