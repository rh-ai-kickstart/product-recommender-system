# Auth Provider Usage Guide

This auth system provides a simple way to manage user authentication using React Query and Context.

## Setup

The `AuthProvider` is already configured in `main.tsx` and wraps your entire app, so you can use the auth hooks anywhere in your component tree.

## Hook

### `useAuth()`

The main hook you'll use for all authentication needs:

```tsx
import { useAuth } from '../contexts/AuthProvider';

function MyComponent() {
  const { user, isLoading, isAuthenticated } = useAuth();

  if (isLoading) return <div>Loading...</div>;

  if (!isAuthenticated) return <div>Please log in</div>;

  return (
    <div>
      <h1>Welcome, {user?.email}!</h1>
      <p>User ID: {user?.user_id}</p>
      <p>Age: {user?.age}</p>
      <p>Gender: {user?.gender}</p>
      <p>Preferences: {user?.preferences}</p>
    </div>
  );
}
```

### Full auth context with login, signup, and logout functions:

```tsx
import { useAuth } from '../contexts/AuthProvider';

function LoginComponent() {
  const { login, signup, logout, isLoading, user, isAuthenticated } = useAuth();

  const handleLogin = async () => {
    try {
      await login({ email: 'user@example.com', password: 'password' });
      // User is now logged in, navigate to dashboard
    } catch (error) {
      console.error('Login failed:', error.message);
    }
  };

  const handleSignup = async () => {
    try {
      await signup({
        email: 'user@example.com',
        password: 'password',
        age: 25,
        gender: 'Other',
      });
      // User is now signed up and logged in
    } catch (error) {
      console.error('Signup failed:', error.message);
    }
  };

  const handleLogout = () => {
    logout();
    // User is now logged out, all cached data is cleared
  };

  return (
    <div>
      {isAuthenticated ? (
        <div>
          <h1>Welcome, {user?.email}!</h1>
          <button onClick={handleLogout}>Logout</button>
        </div>
      ) : (
        <>
          <button onClick={handleLogin} disabled={isLoading}>
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
          <button onClick={handleSignup} disabled={isLoading}>
            {isLoading ? 'Signing up...' : 'Sign Up'}
          </button>
        </>
      )}
    </div>
  );
}
```

## Protected Routes

You can create protected routes by checking authentication status:

```tsx
import { useAuth } from '../contexts/AuthProvider';
import { Navigate } from '@tanstack/react-router';

function ProtectedPage() {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) return <div>Loading...</div>;

  if (!isAuthenticated) {
    return <Navigate to='/login' />;
  }

  return <div>This is a protected page!</div>;
}
```

## What useAuth provides

```tsx
const {
  // User data
  user, // Current user object or null
  isLoading, // Loading state for any auth operation
  isAuthenticated, // Boolean indicating if user is logged in

  // Auth actions
  login, // Function to log in user
  signup, // Function to sign up user
  logout, // Function to log out user
  refetchUser, // Function to refetch user data
} = useAuth();
```

## Features

- **Automatic token management**: Tokens are stored in localStorage and automatically included in API requests
- **Token expiration handling**: Expired tokens are automatically detected and users are logged out
- **React Query integration**: User data is cached and automatically refetched when needed
- **Loading states**: Built-in loading states for all auth operations
- **Error handling**: Comprehensive error handling with meaningful error messages
- **Type safety**: Full TypeScript support with properly typed interfaces

## API Endpoints

The auth system expects these backend endpoints:

- `POST /api/auth/login` - Login with email/password
- `POST /api/auth/signup` - Sign up with email, password, age, gender
- The system stores user data locally after login/signup since there's no `/auth/me` endpoint

## Storage

The auth system uses localStorage to store:

- `auth_token`: JWT token for authentication
- `user_data`: User information (stored after login/signup)

Both are automatically cleared on logout or token expiration.
