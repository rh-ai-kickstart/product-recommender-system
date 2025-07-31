# Services Documentation

This directory contains all API service functions organized by feature area to match the backend routes.

## Directory Structure

```
services/
├── api.ts              # Shared API utilities
├── auth.ts             # Authentication services
├── cart.ts             # Shopping cart operations
├── feedback.ts         # User feedback submission
├── interactions.ts     # User interaction logging
├── orders.ts           # Order management
├── preferences.ts      # User preference management
├── products.ts         # Product search and details
├── recommendations.ts  # Product recommendations
└── wishlist.ts         # Wishlist management
```

## Authentication Service (`auth.ts`)

### Key Features

- **JWT Token Management**: Secure token storage and validation
- **Backend Integration**: Uses `/auth/me` endpoint for proper validation
- **Centralized Validation**: `validateToken()` utility prevents duplication

### API Endpoints

- `POST /auth/login` - User authentication
- `POST /auth/signup` - User registration
- `GET /auth/me` - Get current user (server-side validation)

### Usage

```typescript
import { authService, validateToken } from '../services/auth';

// Login
const authResponse = await authService.login({ email, password });

// Get current user (validates token against backend)
const user = await authService.getCurrentUser();

// Check authentication status
const isValid = authService.isAuthenticated();

// Token validation utility
const { isValid, shouldRedirect } = validateToken();
```

### Token Flow

1. **Login/Signup**: Receive JWT token from backend
2. **Storage**: Token stored in `localStorage` as `auth_token`
3. **Validation**: Token validated against `/auth/me` endpoint
4. **Expiration**: Automatic cleanup on token expiry
5. **Logout**: Clear token and user data

## API Request Utility (`api.ts`)

Shared utility for all API calls with:

- Automatic auth header injection
- Centralized error handling
- Request/response logging
- Type-safe responses

```typescript
import { apiRequest } from './api';

// Usage in service functions
const data = await apiRequest<ProductData[]>('/api/products', 'fetchProducts');
```

## Error Handling

All services use consistent error handling:

- **4xx errors**: Client errors (validation, auth) - no retry
- **5xx errors**: Server errors - automatic retry with backoff
- **Network errors**: Retry with exponential backoff
- **Auth errors**: Automatic token cleanup and redirect

## Service Organization

Each service file corresponds to a backend route:

- `auth.ts` ↔ `backend/routes/auth.py`
- `products.ts` ↔ `backend/routes/products.py`
- `cart.ts` ↔ `backend/routes/cart.py`
- etc.

This 1:1 mapping makes the codebase easy to navigate and maintain.
