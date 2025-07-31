# Contributing Guide

Welcome to the Product Recommender System! This guide will help you set up your development environment and understand our development workflow.

## Development Setup

### Prerequisites

- **Python 3.9+** for backend development
- **Node.js 18+** and **npm** for frontend development
- **PostgreSQL** with pgvector extension
- **Git** for version control

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/product-recommender-system.git
cd product-recommender-system

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python init_backend.py

# Setup frontend (in a new terminal)
cd frontend
npm install

# Start development servers
# Terminal 1: Backend
cd backend && uvicorn main:app --reload

# Terminal 2: Frontend  
cd frontend && npm run dev
```

Your development environment will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Project Structure

```
product-recommender-system/
├── frontend/           # React TypeScript application
│   ├── src/
│   │   ├── components/ # Reusable UI components
│   │   ├── routes/     # Application routing
│   │   ├── hooks/      # Custom React hooks
│   │   ├── services/   # API client services
│   │   └── types/      # TypeScript definitions
│   └── package.json
├── backend/            # FastAPI Python application
│   ├── routes/         # API route handlers
│   ├── services/       # Business logic and ML services
│   ├── database/       # Database models and migrations
│   ├── tests/          # Backend test suite
│   └── requirements.txt
├── helm/               # Kubernetes deployment
└── figures/            # Architecture diagrams
```

## Development Workflow

### 1. Backend Development

**Key Technologies:**
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - Database ORM
- **Feast** - Feature store for ML
- **pgvector** - Vector similarity search
- **Pydantic** - Data validation

**Code Structure:**
```python
# Example API endpoint
from fastapi import APIRouter, Depends
from services.recommendations import RecommendationService

router = APIRouter()

@router.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    service: RecommendationService = Depends()
):
    return await service.get_user_recommendations(user_id)
```

**Running Tests:**
```bash
cd backend
pytest tests/
```

### 2. Frontend Development

**Key Technologies:**
- **React 18** with TypeScript
- **TanStack Router** - Type-safe routing
- **TanStack Query** - Server state management
- **Vite** - Build tool and dev server

**Code Structure:**
```typescript
// Example custom hook
export function useRecommendations(userId: string) {
  return useQuery({
    queryKey: ['recommendations', userId],
    queryFn: () => api.recommendations.getForUser(userId),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
```

**Development Commands:**
```bash
cd frontend

# Start development server
npm run dev

# Run type checking
npm run type-check

# Build for production
npm run build

# Run linting
npm run lint
```

### 3. ML Model Development

**Feature Store Integration:**
```python
# Working with Feast features
from feast import FeatureStore

store = FeatureStore(repo_path=".")
features = store.get_online_features(
    features=["user_features:age", "item_features:category"],
    entity_rows=[{"user_id": "123", "item_id": "456"}]
).to_dict()
```

**Vector Search:**
```python
# pgvector similarity search
from database.db import get_db
from sqlalchemy import text

async def find_similar_items(item_embedding: List[float], limit: int = 10):
    query = text("""
        SELECT item_id, embedding <-> :embedding as distance
        FROM item_embeddings
        ORDER BY distance
        LIMIT :limit
    """)
    return await db.execute(query, {
        "embedding": item_embedding,
        "limit": limit
    })
```

## Testing

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/routes/test_recommendations.py

# Run with verbose output
pytest -v
```

### Frontend Tests

```bash
cd frontend

# Run unit tests (when implemented)
npm test

# Run E2E tests (when implemented)
npm run test:e2e
```

## Code Style and Standards

### Backend (Python)

- **Formatting:** Use `black` for code formatting
- **Linting:** Use `ruff` for linting
- **Type hints:** Use type hints throughout the codebase
- **Docstrings:** Follow Google-style docstrings

```python
# Example code style
async def get_user_recommendations(
    user_id: str,
    limit: int = 10,
    category_filter: Optional[str] = None
) -> List[RecommendationResponse]:
    """Get personalized recommendations for a user.
    
    Args:
        user_id: The user identifier
        limit: Maximum number of recommendations to return
        category_filter: Optional category to filter by
        
    Returns:
        List of recommendation objects
    """
    # Implementation here
```

### Frontend (TypeScript)

- **Formatting:** Use Prettier for code formatting
- **Linting:** ESLint with TypeScript rules
- **Components:** Use functional components with hooks
- **Naming:** PascalCase for components, camelCase for functions

```typescript
// Example component structure
interface ProductCardProps {
  product: Product;
  onAddToCart: (productId: string) => void;
}

export function ProductCard({ product, onAddToCart }: ProductCardProps) {
  const handleAddToCart = () => {
    onAddToCart(product.id);
  };

  return (
    <div className="product-card">
      {/* Component implementation */}
    </div>
  );
}
```

## Database Development

### Schema Changes

1. **Create migration:**
   ```bash
   cd backend
   alembic revision --autogenerate -m "Description of changes"
   ```

2. **Apply migration:**
   ```bash
   alembic upgrade head
   ```

3. **Test with sample data:**
   ```bash
   python init_backend.py
   ```

### Vector Operations

When working with embeddings and vector search:

```sql
-- Create vector index for performance
CREATE INDEX ON item_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Example similarity search
SELECT item_id, embedding <-> '[0.1,0.2,0.3]' as distance
FROM item_embeddings
ORDER BY distance
LIMIT 10;
```

## Deployment Testing

### Local Kubernetes Testing

```bash
# Test Helm chart
cd helm/
helm template product-recommender-system . --values values.yaml

# Validate Kubernetes manifests
kubectl apply --dry-run=client -f <manifest-file>
```

### Integration Testing

```bash
# Test full deployment locally
cd helm/
make install NAMESPACE=test-recommender
make status NAMESPACE=test-recommender
make uninstall NAMESPACE=test-recommender
```

## Submitting Changes

### 1. Branch Naming

Follow conventional branch naming as described in the [conventional branch spec](https://conventional-branch.github.io/#summary):

**Format:** `<type>/<description>`

**Types:**
- `main` - The main development branch
- `feature/` - For new features
- `bugfix/` - For bug fixes
- `hotfix/` - For urgent fixes that need immediate attention
- `release/` - For branches preparing a release
- `chore/` - For non-code tasks like dependency updates, documentation

**Examples:**
```bash
# Feature development
git checkout -b feature/user-authentication
git checkout -b feature/semantic-search-ui

# Bug fixes
git checkout -b bugfix/cart-total-calculation
git checkout -b bugfix/database-connection-timeout

# Urgent fixes
git checkout -b hotfix/security-patch
git checkout -b hotfix/critical-memory-leak

# Release preparation
git checkout -b release/v1.2.0
git checkout -b release/v2.0.0-beta

# Maintenance tasks
git checkout -b chore/update-dependencies
git checkout -b chore/update-documentation
```

**Guidelines:**
- Use lowercase with hyphens (kebab-case) for descriptions
- Keep descriptions concise but descriptive
- Use present tense: `feature/add-wishlist` not `feature/added-wishlist`
- Avoid unnecessary words: `feature/user-auth` not `feature/add-user-authentication-feature`

### 2. Commit Messages

Follow conventional commit message format as described in the [conventional commits spec](https://www.conventionalcommits.org/en/v1.0.0/#summary):
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks
- etc.

### 3. Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure all tests pass** locally
4. **Create pull request** with clear description
5. **Address review feedback** promptly

### Common Development Issues

**Backend server won't start:**
- Check Python virtual environment is activated
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Ensure PostgreSQL is running and accessible

**Frontend build errors:**
- Clear node_modules: `rm -rf node_modules && npm install`
- Check Node.js version: `node --version` (should be 18+)
- Verify TypeScript configuration

**Database connection issues:**
- Check DATABASE_URL environment variable
- Ensure PostgreSQL has pgvector extension installed
- Verify database exists and is accessible

Thank you for contributing to the Product Recommender System! 🚀