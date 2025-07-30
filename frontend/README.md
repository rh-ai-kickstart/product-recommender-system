# Frontend - React TypeScript Application

Modern React application with TypeScript, providing an intuitive interface for product recommendations, search, and e-commerce functionality.

## Quick Start

### Development Setup

```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:5173`

### Project Structure

```
src/
├── components/         # Reusable UI components
│   ├── Carousel/      # Product carousels and galleries
│   ├── product-card.tsx
│   ├── search.tsx
│   └── ...
├── routes/            # Application routing (TanStack Router)
├── hooks/             # Custom React hooks for API integration
├── services/          # API service layer
├── contexts/          # React context providers
├── types/             # TypeScript type definitions
└── utils/             # Utility functions and logging
```

## Key Features

### 🔐 Authentication & User Management
- JWT-based authentication with secure session management
- User registration and login flows
- Account preferences and profile management

### 🛍️ Product Discovery
- **Product Catalog** - Browse and filter products
- **Semantic Search** - Text-based search with AI understanding
- **Image Search** - Upload images to find similar products
- **Recommendations** - Personalized product suggestions

### 🛒 E-commerce Functionality
- **Shopping Cart** - Add, remove, and manage cart items
- **Wishlist** - Save products for later
- **Order Management** - Purchase history and order tracking
- **Product Reviews** - Submit and view product feedback

## Development

### Available Scripts

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linting
npm run lint

# Type checking
npm run type-check
```

### API Integration

The frontend connects to the FastAPI backend using custom hooks and services:

```typescript
// Example: Get personalized recommendations
import { useRecommendations } from './hooks/useRecommendations';

const { data: recommendations, isLoading } = useRecommendations({
  userId: currentUser.id,
  limit: 10
});
```

### Architecture

- **TanStack Router** - Type-safe routing with file-based route structure
- **React Query** - Server state management and caching
- **Context API** - Global state management for authentication and cart
- **TypeScript** - Full type safety throughout the application
- **Vite** - Fast development and optimized builds

## Deployment

The frontend is containerized and deployed alongside the backend in the Kubernetes cluster. See the [main README](../README.md) for full deployment instructions.
