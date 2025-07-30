# Product Recommender System – root-level Makefile
# Run `make help` to see available targets.

.PHONY: help dev backend frontend stop dev-deps backend-deps frontend-deps \
        build build-frontend build-backend lint lint-backend lint-frontend test test-backend \
        compose-up compose-down image-build image-build-frontend image-build-backend \
        install install-help install-status helm-deps uninstall install-namespace clean

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
help: ## Show comprehensive help for all available targets
	@echo "Product Recommender System - Available Make Targets"
	@echo "=================================================="
	@echo ""
	@echo "🚀 Development Commands:"
	@grep -E '^(dev|backend|frontend|stop):.*?## ' $(MAKEFILE_LIST) | \
		sed -e 's/^/  /' -e 's/:.*## / - /'
	@echo ""
	@echo "📦 Dependencies & Setup:"
	@grep -E '^(dev-deps|backend-deps|frontend-deps|clean):.*?## ' $(MAKEFILE_LIST) | \
		sed -e 's/^/  /' -e 's/:.*## / - /'
	@echo ""
	@echo "🔨 Build Commands:"
	@grep -E '^(build|build-frontend|build-backend):.*?## ' $(MAKEFILE_LIST) | \
		sed -e 's/^/  /' -e 's/:.*## / - /'
	@echo ""
	@echo "🧪 Testing & Quality:"
	@grep -E '^(test|test-backend|lint|lint-backend|lint-frontend):.*?## ' $(MAKEFILE_LIST) | \
		sed -e 's/^/  /' -e 's/:.*## / - /'
	@echo ""
	@echo "🐳 Container Commands:"
	@grep -E '^(compose-up|compose-down|image-build|image-build-frontend|image-build-backend):.*?## ' $(MAKEFILE_LIST) | \
		sed -e 's/^/  /' -e 's/:.*## / - /'
	@echo ""
	@echo "☸️  Deployment Commands:"
	@grep -E '^(install|install-help|install-status|helm-deps|uninstall):.*?## ' $(MAKEFILE_LIST) | \
		sed -e 's/^/  /' -e 's/:.*## / - /'
	@echo ""
	@echo "For detailed deployment help, run: make install-help"

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------

dev: ## Run backend and frontend development servers together
	@echo "Starting development environment..."
	@echo "Backend will be available at: http://localhost:8000"
	@echo "Frontend will be available at: http://localhost:5173"
	@echo "Starting backend in background..."
	@cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
	@echo "Starting frontend..."
	@cd frontend && npm run dev

backend: ## Run backend dev server with hot-reload (uvicorn)
	@echo "Starting backend development server at http://localhost:8000"
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

frontend: ## Run frontend Vite dev server
	@echo "Starting frontend development server at http://localhost:5173"
	cd frontend && npm run dev

stop: ## Stop all background development processes
	@echo "Stopping development servers..."
	@pkill -f "uvicorn main:app" || true
	@pkill -f "vite" || true

# -----------------------------------------------------------------------------
# Build helpers (build without installing)
# -----------------------------------------------------------------------------
build-frontend: ## Build frontend application for production
	@echo "Building frontend for production..."
	cd frontend && npm run build

build-backend: ## Validate backend can be imported/started
	@echo "Validating backend setup..."
	cd backend && python -c "from main import app; print('Backend validation successful')"

build: build-frontend build-backend ## Build both frontend and backend

# -----------------------------------------------------------------------------
# Installation helpers
# -----------------------------------------------------------------------------
dev-deps: backend-deps frontend-deps ## Install backend & frontend development dependencies

backend-deps: ## Install backend dependencies with pip
	@echo "Installing backend dependencies..."
	cd backend && pip install -e ".[dev]"

frontend-deps: ## Install Node dependencies
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

# -----------------------------------------------------------------------------
# Quality & Tests
# -----------------------------------------------------------------------------
lint-backend: ## Run backend linters (ruff)
	@echo "Running backend linting..."
	cd backend && ruff check .

lint-frontend: ## Run frontend linters (eslint)
	@echo "Running frontend linting..."
	cd frontend && npm run lint

lint: lint-backend lint-frontend ## Run all linters

test-backend: ## Run backend tests with pytest
	@echo "Running backend tests..."
	cd backend && python -m pytest --cov=. --cov-report=term-missing

test: lint test-backend ## Run full test & lint suite

# -----------------------------------------------------------------------------
# Container / Compose helpers
# -----------------------------------------------------------------------------
compose-up: ## Start services with podman compose
	@echo "Starting services with podman compose..."
	podman compose up -d

compose-down: ## Stop compose services
	@echo "Stopping compose services..."
	podman compose down

image-build-frontend: ## Build frontend container image
	@echo "Building frontend container image..."
	cd frontend && podman build -t product-recommender-frontend:dev .

image-build-backend: ## Build backend container image
	@echo "Building backend container image..."
	cd backend && podman build -f Containerfile -t product-recommender-backend:dev .

image-build: image-build-frontend image-build-backend ## Build all container images

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
clean: ## Clean build artifacts and dependencies
	@echo "Cleaning build artifacts..."
	rm -rf frontend/dist frontend/node_modules
	rm -rf backend/__pycache__ backend/.pytest_cache backend/.coverage
	rm -rf ui/dist ui/node_modules
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# -----------------------------------------------------------------------------
# Deployment (Helm/OpenShift)
# -----------------------------------------------------------------------------

# Helm deployment configuration variables
NAMESPACE ?=
POSTGRES_USER ?= postgres
POSTGRES_PASSWORD ?= recsys_password
POSTGRES_DBNAME ?= recsys
PRODUCT_RECOMMENDER_CHART := product-recommender-system
TOLERATIONS_TEMPLATE=[{"key":"$(1)","effect":"NoSchedule","operator":"Exists"}]

# Check namespace is set only for deployment commands that need it
ifneq (,$(filter install install-status uninstall install-namespace,$(MAKECMDGOALS)))
ifeq ($(NAMESPACE),)
$(error NAMESPACE is not set. Use: make <target> NAMESPACE=<your-namespace>)
endif
endif

# Helm argument builders
helm_pgvector_args = \
    --set pgvector.secret.user=$(POSTGRES_USER) \
    --set pgvector.secret.password=$(POSTGRES_PASSWORD) \
    --set pgvector.secret.dbname=$(POSTGRES_DBNAME)

dataset_args = \
	--set datasetUrl=$(DATASET_URL)

install-help: ## Show detailed deployment help and configuration options
	@echo "Product Recommender System Deployment Help"
	@echo "=========================================="
	@echo ""
	@echo "Available deployment targets:"
	@echo "  install          - Install the Product Recommender System deployment"
	@echo "  install-status   - Check deployment status"
	@echo "  uninstall        - Uninstall deployment and clean up resources"
	@echo "  helm-deps        - Update Helm dependencies"
	@echo ""
	@echo "Required Configuration:"
	@echo "  NAMESPACE        - Target namespace (required for all deployment commands)"
	@echo ""
	@echo "Optional Configuration (set via environment variables or make arguments):"
	@echo "  POSTGRES_USER     - PostgreSQL username (default: postgres)"
	@echo "  POSTGRES_PASSWORD - PostgreSQL password (default: recsys_password)"
	@echo "  POSTGRES_DBNAME   - PostgreSQL database name (default: recsys)"
	@echo "  DATASET_URL       - URL for dataset loading"
	@echo ""
	@echo "Example usage:"
	@echo "  make install NAMESPACE=my-recommender-system"
	@echo "  make install-status NAMESPACE=my-recommender-system"

helm-deps: ## Update Helm dependencies
	@echo "Updating Helm dependencies"
	@cd helm && helm dependency update $(PRODUCT_RECOMMENDER_CHART) &> /dev/null

install-namespace: ## Create and configure deployment namespace
	@oc create namespace $(NAMESPACE) &> /dev/null && oc label namespace $(NAMESPACE) modelmesh-enabled=false ||:
	@oc project $(NAMESPACE) &> /dev/null ||:

install: install-namespace helm-deps ## Install the Product Recommender System deployment
	@$(eval PGVECTOR_ARGS := $(call helm_pgvector_args))
	@$(eval DATASET_ARGS := $(call dataset_args))
	@echo "Installing $(PRODUCT_RECOMMENDER_CHART) helm chart in namespace $(NAMESPACE)"
	@cd helm && helm upgrade --install $(PRODUCT_RECOMMENDER_CHART) $(PRODUCT_RECOMMENDER_CHART) -n $(NAMESPACE) \
		$(PGVECTOR_ARGS) \
		$(DATASET_ARGS) \
		$(EXTRA_HELM_ARGS) --set strimzi-kafka-operator.createGlobalResources=false --timeout 300m
	@echo "Waiting for model services and deployment to complete. It may take around 10-15 minutes..."
	@oc rollout status deploy -n $(NAMESPACE)
	@echo "$(PRODUCT_RECOMMENDER_CHART) installed successfully"
	@echo ""
	@echo "Getting application URL..."
	@sleep 5
	@$(eval APP_URL := $(shell oc get routes product-recommender-frontend -n $(NAMESPACE) -o jsonpath='{.status.ingress[0].host}' 2>/dev/null || echo ""))
	@if [ -n "$(APP_URL)" ]; then \
		echo "🎉 Application is ready!"; \
		echo "📱 Access your Product Recommender System at: https://$(APP_URL)"; \
	else \
		echo "⚠️  Route not ready yet. Get the URL manually with:"; \
		echo "   oc get routes product-recommender-frontend -n $(NAMESPACE)"; \
	fi
	@echo ""

uninstall: ## Uninstall deployment and clean up resources
	@echo "Uninstalling $(PRODUCT_RECOMMENDER_CHART) helm chart from namespace $(NAMESPACE)"
	@cd helm && helm uninstall --ignore-not-found $(PRODUCT_RECOMMENDER_CHART) -n $(NAMESPACE)
	@echo "Removing pgvector and minio PVCs from $(NAMESPACE)"
	@oc get pvc -n $(NAMESPACE) -o custom-columns=NAME:.metadata.name | grep -E '^(pg|minio)-data' | xargs -I {} oc delete pvc -n $(NAMESPACE) {} ||:
	@echo "Deleting remaining pods in namespace $(NAMESPACE)"
	@oc delete jobs -n $(NAMESPACE) --all
	@oc delete pods -n $(NAMESPACE) --all
	@echo "Checking for any remaining resources in namespace $(NAMESPACE)..."
	@echo "If you want to completely remove the namespace, run: oc delete project $(NAMESPACE)"
	@echo "Remaining resources in namespace $(NAMESPACE):"
	@$(MAKE) install-status NAMESPACE=$(NAMESPACE)

install-status: ## Check deployment status
	@echo "Deployment status for namespace: $(NAMESPACE)"
	@echo "========================================"
	@echo ""
	@echo "Pods:"
	@oc get pods -n $(NAMESPACE) || true
	@echo ""
	@echo "Services:"
	@oc get svc -n $(NAMESPACE) || true
	@echo ""
	@echo "Routes:"
	@oc get routes -n $(NAMESPACE) || true
	@echo ""
	@echo "Secrets:"
	@oc get secrets -n $(NAMESPACE) | grep huggingface-secret || true
	@echo ""
	@echo "PVCs:"
	@oc get pvc -n $(NAMESPACE) || true