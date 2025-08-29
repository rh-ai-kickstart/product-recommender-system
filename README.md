# AI Quickstart - Product Recommender System

Welcome to the Product Recommender System Quickstart!
Use this to quickly get a recommendation engine with user-item relationships up and running in your environment.

To see how it's done, jump straight to [installation](#install).

## 🎯 Description
The Product Recommender System Quickstart enables the rapid establishment of a scalable and personalized product recommendation service.

The system recommends items to users based on their previous interactions with products and the behavior of similar users.

It supports recommendations for both existing and new users. New users are prompted to select their preferences to personalize their experience.

Users can interact with the user interface to view items, add items to their cart, make purchases, or submit reviews.

### Main Features
To find products in the application you can do a:
* Scroll recommended items.
* Search items by text (semantic search).
* Search items by Image (find similar items in the store).

## 📁 Project Structure

This repository contains multiple components that work together to create a complete recommendation system:

### 🏗️ Core Components

#### [recommendation-core/](recommendation-core/README.md)
**ML Library & Models** - The foundational Python library for building recommendation systems
- **Multi-modal neural networks**: Two-tower architecture for users and items with support for text, image, and numerical features
- **Feature processing**: Comprehensive data preprocessing for text, image, numerical, and categorical data handling
- **CLIP integration**: Advanced text and image feature extraction using OpenAI's CLIP model for semantic understanding
- **Data generation**: Synthetic datasets creation and AI-powered product image generation using Stable Diffusion
- **Search services**: High-performance text and image-based similarity search with vector embeddings
- **Model serving**: Real-time inference capabilities for recommendation generation
- **Data utilities**: Tools for data validation, transformation, and feature engineering

#### [recommendation-training/](recommendation-training/README.md)
**Training Pipeline** - Kubeflow-based ML training workflows for production-ready model training
- **Data loading**: Feast feature store integration for consistent feature serving across training and inference
- **Model training**: Two-tower neural network training with configurable architectures and hyperparameters
- **Model registration**: Model Registry integration for versioning, tracking, and deployment management
- **Candidate generation**: Embedding generation and online serving for real-time recommendations
- **Batch scoring**: Automated recommendation generation and storage for offline serving
- **Pipeline orchestration**: Kubeflow Pipelines for reproducible ML workflows
- **Cluster operations**: OpenShift integration for credential management and service discovery
- **Monitoring**: Training metrics, model performance tracking, and pipeline observability



### 🚀 Deployment Components

#### [backend/](backend/)
**FastAPI Backend** - High-performance REST API and business logic layer
- **User authentication**: JWT-based authentication system with secure token management
- **Product management**: Complete CRUD operations for products, users, and interactions
- **Recommendation serving**: Real-time recommendation API with caching and optimization
- **Feast integration**: Feature store for online serving with real-time feature updates
- **Static file serving**: Frontend asset delivery and CDN-like functionality
- **Database integration**: PostgreSQL with pgvector for vector similarity search
- **API documentation**: Auto-generated OpenAPI/Swagger documentation
- **Error handling**: Comprehensive error management and logging
- **Rate limiting**: API rate limiting and request throttling
- **Health checks**: System health monitoring and status endpoints

#### [frontend/](frontend/)
**React Frontend** - Modern, responsive user interface and interactions
- **Product browsing**: Interactive item catalog with filtering and sorting capabilities
- **User preferences**: Intelligent preference selection for new users with guided onboarding
- **Shopping cart**: Full e-commerce cart functionality with add/remove items
- **Recommendation display**: Personalized item recommendations with real-time updates
- **Image search**: Visual similarity search capabilities with drag-and-drop interface
- **Text search**: Semantic search with autocomplete and search suggestions
- **User dashboard**: Personal user dashboard with purchase history and preferences
- **Responsive design**: Mobile-first design with cross-device compatibility
- **Performance optimization**: Lazy loading, caching, and optimized bundle delivery
- **Accessibility**: WCAG compliant interface with screen reader support

#### [helm/](helm/)
**Kubernetes Deployment** - Complete system orchestration and infrastructure management
- **Helm charts**: All components packaged for easy deployment with configurable values
- **Service mesh**: Istio-based traffic management with advanced routing and security
- **Database setup**: PostgreSQL and pgvector configuration with automated initialization
- **Monitoring**: OpenShift built-in monitoring stack (Prometheus/Thanos + Grafana) for comprehensive observability
- **Security**: RBAC, network policies, and security context constraints
- **Auto-scaling**: Horizontal Pod Autoscaler (HPA) for dynamic resource management
- **Ingress configuration**: Load balancing and SSL termination setup
- **Resource management**: CPU and memory limits with resource quotas
- **Backup and recovery**: Automated backup strategies for data persistence
- **Multi-environment support**: Development, staging, and production configurations

### 📊 Data & Storage

#### [figures/](figures/)
**Architecture Diagrams** - Comprehensive system design documentation
- **Data processing pipeline**: End-to-end data flow from ingestion to serving
- **Training workflow**: Detailed ML model training process and pipeline stages
- **Inference architecture**: Real-time recommendation serving and caching strategies
- **Search capabilities**: Text and image search flows with vector similarity
- **System integration**: Component interaction and data flow visualization
- **Deployment topology**: Kubernetes resource relationships and networking

### 🔧 Infrastructure Components

#### [tests/](tests/)
**Testing Framework** - Comprehensive testing suite for all system components
- **Unit tests**: Component-level testing for individual functions and classes
- **Integration tests**: End-to-end testing for component interactions
- **Performance tests**: Load testing and performance benchmarking
- **Security tests**: Vulnerability scanning and security validation
- **API tests**: REST API testing with automated test suites
- **UI tests**: Frontend testing with automated browser testing



## 🏗️ Architecture Overview

### System Components

The recommendation system consists of several interconnected components:

1. **Data Generation** (`recommendation-core/generation/`)
   - Synthetic user, item, and interaction data with realistic patterns
   - Product image generation using Stable Diffusion for diverse catalog
   - Realistic e-commerce dataset creation with configurable parameters
   - Data validation and quality assurance processes

2. **Feature Store** (`recommendation-core/feature_repo/`)
   - Feast-based feature management with online/offline serving
   - Real-time feature updates and versioning
   - Feature transformation and preprocessing pipelines
   - Feature monitoring and drift detection

3. **Model Training** (`recommendation-training/`)
   - Two-tower neural network training with configurable architectures
   - Embedding generation for users and items with similarity optimization
   - Model versioning and registration with metadata tracking
   - Automated hyperparameter tuning and model selection

4. **Inference Serving** (`backend/`)
   - Real-time recommendation API with sub-second response times
   - Similarity search capabilities with vector database integration
   - User preference management and personalization
   - Caching strategies for improved performance

5. **User Interface** (`frontend/`)
   - Product browsing and search with advanced filtering
   - User preference selection with intelligent recommendations
   - Shopping cart functionality with persistent state
   - Responsive design with mobile optimization

### Data Flow

```
Data Generation → Feature Store → Model Training → Model Registry → Inference Serving → User Interface
```

### Component Interactions

1. **Data Pipeline**: Raw data flows through the feature store for consistent feature serving
2. **Training Pipeline**: Models are trained using Kubeflow Pipelines with automated workflows
3. **Model Registry**: Trained models are versioned and stored for deployment
4. **Inference Engine**: Real-time recommendations are served through the backend API
5. **User Interface**: Frontend provides interactive access to recommendations and search

## 🔄 Complete System Workflow

### Overview
The Product Recommender System follows a comprehensive workflow from data generation to user interaction. This section walks through the entire process, from creating synthetic datasets to users adding items to their cart.

### Phase 1: Data Generation & Preparation

#### 1.1 Synthetic Dataset Creation
```bash
# Generate Amazon-like e-commerce data
python -m recommendation_core.generation.dataset_gen_amazon \
    --n_users 1000 \
    --n_items 5000 \
    --n_interactions 20000
```

**What happens:**
- **User profiles** are generated with realistic demographics and preferences
- **Product catalog** is created with categories, prices, and descriptions
- **User-item interactions** are simulated with realistic patterns
- **Parquet files** are saved to `src/recommendation_core/feature_repo/data/`

**When dataset_gen_amazon.py is Run:**

**Manual Execution by Developers** (Primary Method)
Developers run it manually during development and setup:

**When this happens:**
- **Initial setup** - When setting up the project for the first time
- **Development** - When developers need fresh test data
- **Testing** - When testing the recommendation system with different dataset sizes
- **Demo preparation** - When preparing demos with specific data volumes

**No Automated Generation**
There's **no automated process** that runs `dataset_gen_amazon.py`:
- ❌ No CI/CD pipeline runs it
- ❌ No container startup script runs it
- ❌ No training pipeline runs it
- ✅ Only manual execution by developers

**Training Pipeline Uses Pre-Generated Data**
The training pipeline **doesn't run** `dataset_gen_amazon.py` directly. Instead, it uses **pre-generated parquet files**:

```python
# From train-workflow.py - load_data_from_feast()
if dataset_url is not None and dataset_url != "":
    logger.info("using custom remote dataset")
    dataset_provider = RemoteDatasetProvider(dataset_url, force_load=True)
else:
    logger.info("using pre generated dataset")  # ← Uses existing parquet files
    dataset_provider = LocalDatasetProvider(store)
```

**What this means:**
- The training pipeline expects the parquet files to **already exist**
- It loads data from `src/recommendation_core/feature_repo/data/`
- The files are created **before** the training pipeline runs

**Why This Design:**
- **Reproducible datasets** - Same data across all environments
- **Fast training** - No need to generate data during training
- **Version control** - Dataset files can be committed to git
- **Consistent testing** - Same test data for all tests

**Parquet File Creation Process:**

**Location:** All parquet files are written to:
```
src/recommendation_core/feature_repo/data/
```

**Files Generated:**
The `dataset_gen_amazon.py` script creates **9 parquet files** (binary columnar format) with the following structure:

**Main Dataset Files:**
1. `recommendation_users.parquet` - User profiles with demographics and preferences
2. `recommendation_items.parquet` - Product catalog with features and pricing
3. `recommendation_interactions.parquet` - User-item interaction history

**Dummy/Feature Files:**
4. `dummy_item_embed.parquet` - Placeholder item embeddings for feature store
5. `dummy_user_embed.parquet` - Placeholder user embeddings for feature store
6. `user_items.parquet` - User-item relationship mappings
7. `item_textual_features_embed.parquet` - Text feature embeddings
8. `item_clip_features_embed.parquet` - CLIP image feature embeddings

**Code Implementation:**
```python
# From dataset_gen_amazon.py - Lines 265-315
data_path = pathlib.Path("src/recommendation_core/feature_repo/data")
data_path.mkdir(parents=True, exist_ok=True)

# Save main datasets
users.to_parquet(data_path / "recommendation_users.parquet", index=False)
items.to_parquet(data_path / "recommendation_items.parquet", index=False)
interactions.to_parquet(data_path / "recommendation_interactions.parquet", index=False)

# Save dummy/feature files
dummy_item_embed_df.to_parquet(data_path / "dummy_item_embed.parquet", index=False)
dummy_user_embed_df.to_parquet(data_path / "dummy_user_embed.parquet", index=False)
dummy_user_items_df.to_parquet(data_path / "user_items.parquet", index=False)
dummy_textual_feature_df.to_parquet(data_path / "item_textual_features_embed.parquet", index=False)
dummy_clip_feature_df.to_parquet(data_path / "item_clip_features_embed.parquet", index=False)
```

**Data Generation Details:**

**User Data (`recommendation_users.parquet`):**
- **User IDs**: 26-character alphanumeric strings
- **User Names**: "Customer" prefix with unique identifiers
- **Signup Dates**: Random dates within 2023
- **Preferences**: 1-5 categories per user (e.g., "Electronics|Computers&Accessories")

**Item Data (`recommendation_items.parquet`):**
- **Item IDs**: Amazon-style B0XXXXXXXX format
- **Product Names**: Realistic product names (e.g., "WiFi Ultra AC1200")
- **Categories**: Hierarchical categories (e.g., "Computers&Accessories|NetworkingDevices|Routers")
- **Pricing**: Actual price, discounted price, and discount percentage
- **Ratings**: Random ratings (1-5 stars) with rating counts
- **Descriptions**: Detailed product descriptions for image generation
- **Image Links**: References to generated product images

**Interaction Data (`recommendation_interactions.parquet`):**
- **Interaction Types**: view, cart, purchase, rate
- **Timestamps**: Realistic interaction timing
- **User-Item Matching**: Biased selection based on user preferences
- **Reviews**: Generated review titles and content for rated items
- **Quantities**: Purchase quantities for buy interactions

**Integration with Image Generation:**
The dataset generation script **references generated images**:
```python
# From dataset_gen_amazon.py
img_link = f"/images/item_{safe_name}.png"
```

**Directory Structure Created:**
```
recommendation-core/
└── src/
    └── recommendation_core/
        └── feature_repo/
            └── data/                    # ← All parquet files written here
                ├── recommendation_users.parquet
                ├── recommendation_items.parquet
                ├── recommendation_interactions.parquet
                ├── dummy_item_embed.parquet
                ├── dummy_user_embed.parquet
                ├── user_items.parquet
                ├── item_textual_features_embed.parquet
                └── item_clip_features_embed.parquet
```

**Key Features:**
- **Auto-creation**: Directory is created automatically if it doesn't exist
- **Relative paths**: Uses relative paths from execution location
- **Feast integration**: Location aligns with Feast feature store structure
- **Training pipeline**: Files are expected by the training workflow
- **Version control**: Files can be committed to git for reproducibility

**Why These Parquet Files Are Created:**

**1. Synthetic Data for Development & Testing**
- **No real data dependency**: Eliminates need for actual e-commerce data
- **Controlled environment**: Predictable data patterns for testing
- **Privacy compliance**: No PII or sensitive business data
- **Reproducible results**: Same data across all environments

**2. Machine Learning Model Training**
- **Feature engineering**: Structured data for neural network training
- **User-item interactions**: Training data for collaborative filtering
- **Multi-modal features**: Text, image, and numerical data for advanced models
- **Realistic patterns**: Simulated user behavior for model validation

**3. Feast Feature Store Integration**
- **Online serving**: Real-time feature access for inference
- **Offline training**: Batch feature serving for model training
- **Feature versioning**: Consistent feature definitions across environments
- **Scalable architecture**: Supports both online and offline serving

**4. Demo & Presentation Purposes**
- **Complete e-commerce experience**: Full product catalog with images
- **Realistic user interactions**: Simulated shopping behavior
- **Visual content**: Generated product images for UI
- **Scalable demonstrations**: Configurable dataset sizes

**Where Parquet Files Are Used:**

**1. Training Pipeline (`recommendation-training/`)**
```python
# From train-workflow.py - load_data_from_feast()
dataset_provider = LocalDatasetProvider(store)  # Loads from parquet files
item_df = dataset_provider.item_df()           # recommendation_items.parquet
user_df = dataset_provider.user_df()           # recommendation_users.parquet
interaction_df = dataset_provider.interaction_df()  # recommendation_interactions.parquet
```

**2. Feast Feature Store (`recommendation-core/feature_repo/`)**
```yaml
# feature_store.yaml references parquet files
data_source:
  path: data/recommendation_items.parquet  # Item features
  path: data/recommendation_users.parquet  # User features
  path: data/recommendation_interactions.parquet  # Interaction history
```

**3. Backend API (`backend/`)**
```python
# From feast_service.py - Real-time inference
store = FeatureStore(repo_path="src/recommendation_core/feature_repo/")
dataset_provider = LocalDatasetProvider(store, data_dir="/app/backend/src/services/feast/data")
```

**4. Frontend Display (`frontend/`)**
- **Product catalog**: Items from `recommendation_items.parquet`
- **User preferences**: Data from `recommendation_users.parquet`
- **Recommendation serving**: Results from trained models using parquet data

**5. Model Registry & Deployment**
- **Model training**: Uses parquet files as training data
- **Embedding generation**: Creates user/item embeddings from parquet data
- **Model serving**: Serves recommendations based on parquet-based features

**Data Flow Through the System:**

```
Parquet Files Created → Feast Feature Store → Training Pipeline →
Model Training → Embedding Generation → Model Registry →
Backend API → Frontend Display → User Interactions
```

**Benefits of This Approach:**

**1. Development Efficiency**
- **Quick setup**: No need to source real e-commerce data
- **Consistent environment**: Same data across development, testing, production
- **Rapid iteration**: Easy to regenerate datasets with different parameters

**2. Production Readiness**
- **Scalable architecture**: Can handle real data with same structure
- **Feature store integration**: Ready for online/offline serving
- **Model versioning**: Reproducible training with versioned datasets

**3. Demo & Sales**
- **Complete experience**: Full e-commerce functionality
- **Visual appeal**: Generated product images
- **Realistic interactions**: Simulated user behavior
- **Configurable scale**: Adjustable dataset sizes for different demos

**4. Research & Experimentation**
- **Controlled variables**: Predictable data for A/B testing
- **Feature experimentation**: Easy to modify data structure
- **Model comparison**: Consistent datasets for model evaluation

#### 1.2 AI-Powered Image Generation
```bash
# Generate product images using Stable Diffusion
python -m recommendation_core.generation.generate_images
```

**What happens:**
- **Product descriptions** are used as prompts for Stable Diffusion
- **Synthetic product images** are generated for each item
- **Images are saved** to `src/recommendation_core/generation/data/generated_images/`
- **Dataset references** these images in the `img_link` field

**What generate_images.py Does:**
- **Reads product data** from `src/recommendation_core/feature_repo/data/item_df_output.parquet`
- **Uses product descriptions** as prompts for Stable Diffusion
- **Generates synthetic product images** using AI
- **Saves images** to `src/recommendation_core/generation/data/generated_images/`
- **Creates realistic product visuals** for the recommendation system

**Features:**
- **Stable Diffusion v1.5**: High-quality image generation
- **Text-to-image**: Product descriptions to images
- **GPU acceleration**: CUDA support for faster generation
- **Batch processing**: Efficient multi-image generation

**Integration with Dataset Generation:**
The generated images are **referenced by dataset_gen_amazon.py**:

```python
# From dataset_gen_amazon.py
img_link = f"/images/item_{safe_name}.png"
```

**Current Status:**
- ✅ **99 images generated** and stored in repository
- ✅ **Images actively used** by the recommendation system
- ✅ **Manual execution** - Run when new products are added
- ✅ **One-time setup** - Images are committed to git for reuse

**Data Flow:**
```
1. dataset_gen_amazon.py creates product data with descriptions
         ↓
2. generate_images.py reads product descriptions as prompts
         ↓
3. Stable Diffusion generates images from text descriptions
         ↓
4. Images saved to generated_images/ directory
         ↓
5. dataset_gen_amazon.py references these images in img_link
         ↓
6. Recommendation system displays generated product images
```

**Image Storage Location:**
```
recommendation-core/
└── src/
    └── recommendation_core/
        └── generation/
            └── data/
                └── generated_images/           # ← Images stored here
                    ├── item_AirFlow Maestro.png
                    ├── item_BassBoost Pro.png
                    ├── item_CableMax 8K.png
                    ├── item_DigitalCanvas Pro.png
                    ├── item_Galaxy X23 Ultra.png
                    └── ... (99 total images)
```

**File Naming Convention:**
- **Format**: `item_{ProductName}.png`
- **Example**: `item_WiFi Ultra AC1200.png`
- **URL Encoding**: Spaces replaced with `%20` in dataset references
- **Size**: ~200-600KB per image (high-quality PNG format)

**Why This Design:**
- **Synthetic data** - No need for real product images
- **Consistent style** - All images generated with same AI model
- **Scalable** - Can generate unlimited product images
- **Demo-friendly** - Provides visual content for demonstrations

#### 1.3 Feature Store Setup
```bash
# Apply Feast feature store configuration
feast apply
```

**What happens:**
- **Feature views** are created for users, items, and interactions
- **Online/offline serving** is configured for real-time feature access
- **Feature transformations** are applied for consistent data processing

**Feast's Architecture & Multiple Roles:**

**Data Flow:**
```
Static Parquet Files → Feast Feature Store (PostgreSQL with PGVector extension) → Model Training
```

**Feast with PostgreSQL + PGVector as Data Store:**
- **Feature Schema Management**: Knows about your features and their schemas
- **Data Versioning & Lineage**: Handles data versioning and lineage tracking
- **ML-Optimized APIs**: Provides machine learning optimized APIs
- **Batch & Real-time Serving**: Manages both batch and real-time serving
- **Vector Operations**: Uses PGVector extension for similarity search
- **Scalable Storage**: PostgreSQL handles large datasets efficiently
- **Concurrent Access**: Supports high-throughput operations

**Feast's Multiple Roles in the System:**

**1. Initial Setup (Data Ingestion)**
- **Creates Feature Store** using Parquet files with PostgreSQL backend
- **Sets up schema** and feature definitions
- **Configures feature views** for users, items, and interactions
- **Establishes data lineage** and versioning
- **Feast Apply Job (One-Time Setup)**
  ```yaml
  apiVersion: batch/v1
  kind: Job  # ← Regular Job, NOT CronJob
  metadata:
    name: feast-apply-job
  ```
  - **One-time execution**: Runs only once during initial deployment
  - **Infrastructure setup**: Configures feature store schema and definitions
  - **No recurring schedule**: Not scheduled to run repeatedly
  - **Persistent configuration**: Once set up, feature store remains stable
  - **Prerequisite for training**: Training pipeline waits for this job to complete

**2. Real-time Serving (Production Use)**
Feast is actively used for real-time serving in production:

**3. Real-time Product Recommendations**
```python
# From feast_service.py - Real-time recommendation serving
store = FeatureStore(repo_path="src/recommendation_core/feature_repo/")
user_features = store.get_online_features(features=user_feature_view, entity_rows=[{"user_id": user_id}])
```

**What happens in real-time:**
- **User requests recommendations** → Feast serves pre-computed top-k items
- **Sub-second response times** for instant recommendations
- **Personalized results** based on user preferences and history
- **Scalable serving** for multiple concurrent users

**4. New User Recommendations (Real-time)**
```python
# Embedding-based recommendations for new users
user_embed = user_encoder(**data_preproccess(user_as_df))[0]
top_k = store.retrieve_online_documents(query=user_embed.tolist(), top_k=k)
```

**What happens in real-time:**
- **New user signs up** → User embedding generated on-the-fly
- **Vector similarity search** → PGVector finds similar items instantly
- **Real-time results** → Recommendations served immediately

**5. Real-time Semantic Search**
```python
# Text and image-based similarity search
search_service = SearchService(store)
results_df = search_service.search_by_text(text, k)
```

**What happens in real-time:**
- **User enters search query** → Text embedding generated instantly
- **Vector similarity search** → PGVector finds similar products
- **Real-time results** → Search results served immediately

**6. Training Data Serving (Batch)**
```python
# From train-workflow.py - Batch training data
dataset_provider = LocalDatasetProvider(store)
item_df = dataset_provider.item_df()
user_df = dataset_provider.user_df()
interaction_df = dataset_provider.interaction_df()
```

**7. Real-time Embedding Storage and Retrieval**
```python
# Vector similarity search with PGVector
similar_items = store.retrieve_online_documents(
    query=user_embedding,
    top_k=10,
    features=["item_embedding:item_id"]
)
```

**8. User Management & Profile Data**
```python
# Get all existing users from feature store
def get_all_existing_users(self) -> List[dict]:
    user_df = self.dataset_provider.user_df()
    return user_df
```

**9. Product Catalog & Item Details**
```python
# Fetch full product details from feature store
suggested_item = self.store.get_online_features(
    features=self.store.get_feature_service("item_service"),
    entity_rows=[{"item_id": item_id} for item_id in top_item_ids],
).to_df()
```

**10. Image-Based Product Search**
```python
# Search products by uploaded image
def search_item_by_image_file(self, image: PILImage.Image, k=5):
    results_df = self.search_by_image_service.search_by_image(image, k)
    top_item_ids = results_df["item_id"].tolist()
    return self._item_ids_to_product_list(top_item_ids)
```

**11. Individual Product Lookup**
```python
# Get specific product by ID
def get_item_by_id(self, item_id: int) -> Product:
    product_list = self._item_ids_to_product_list([item_id])
    return product_list[0]
```

**Real-time Recommendation Flow:**
```
User Request → Feast API → PostgreSQL+PGVector → Vector Similarity Search →
Real-time Results → User Interface
```

**Complete Feast Usage Summary:**
- ✅ **Product Recommendations** - Personalized item suggestions
- ✅ **User Management** - User profiles and preferences
- ✅ **Product Catalog** - Item details and metadata
- ✅ **Semantic Search** - Text and image-based search
- ✅ **Training Data** - Batch data for model training
- ✅ **Vector Operations** - Embedding storage and retrieval
- ✅ **Real-time Serving** - Sub-second response times
- ✅ **Feature Versioning** - Consistent feature definitions

**Key Benefits of This Architecture:**
- **Unified Data Access**: Same features for training and inference
- **Real-time Performance**: Sub-second response times for recommendations
- **Scalable Storage**: PostgreSQL handles large datasets efficiently
- **Vector Operations**: PGVector enables fast similarity search
- **Data Consistency**: Versioned features ensure reproducibility

### Phase 2: Model Training & Registration

#### 2.1 Training Pipeline Execution

**Why Models Are Trained Periodically:**

Models are trained **daily at midnight UTC** to address the dynamic nature of e-commerce environments and ensure optimal recommendation performance. **Data drift and concept drift** occur as user preferences evolve over time, new products are added to catalogs, seasonal patterns emerge, and market dynamics shift. Without periodic retraining, models would lose accuracy as they become outdated and fail to capture current user behaviors and product trends. **Continuous learning** is essential because daily user interactions—including clicks, purchases, ratings, and browsing patterns—provide valuable fresh training data that improves model accuracy and enables better cold-start recommendations for new users and items. From a **business perspective**, periodic training directly impacts revenue optimization by generating more relevant recommendations that increase conversion rates, enhances user engagement through improved personalization, provides competitive advantage by staying current with market trends, and boosts customer satisfaction through more accurate suggestions. **Technical benefits** include maintaining model freshness to reflect current data patterns, enabling regular performance monitoring against new data, supporting A/B testing to compare model versions, and providing rollback capabilities for safety. The **data pipeline integration** supports this through streaming data collection of new interactions, daily feature store updates with fresh features, periodic embedding refreshes for users and items, and model registry version control for lifecycle management.
```bash
# Run Kubeflow training pipeline
python train-workflow.py
```

**What happens:**
- **Data loading** from Feast feature store
- **Two-tower model training** (UserTower + ItemTower)
- **Embedding generation** for all users and items
- **Model versioning** and registration in Model Registry

#### 2.2 Batch Scoring & Candidate Generation
```python
# Generate top-k recommendations for all users
for user in users:
    user_embedding = user_encoder(user_features)
    similar_items = similarity_search(user_embedding, top_k=10)
    store_recommendations(user_id, similar_items)
```

**What happens:**
- **User embeddings** are generated using trained UserTower
- **Item embeddings** are generated using trained ItemTower
- **Similarity search** finds top-k items for each user
- **Recommendations** are stored in online feature store

### Phase 3: System Deployment

#### 3.1 Infrastructure Setup
```bash
# Deploy using Helm charts
helm install product-recommender ./helm
```

**What happens:**
- **PostgreSQL + pgvector** database is deployed
- **Backend API** (FastAPI) is deployed with Feast integration
- **Frontend** (React) is deployed with static file serving
- **Monitoring** (OpenShift built-in Prometheus/Thanos + Grafana) is configured

#### 3.2 Model Deployment
```bash
# Deploy trained models to serving infrastructure
kubectl apply -f model-deployment.yaml
```

**What happens:**
- **Model artifacts** are loaded from Model Registry
- **Inference endpoints** are created for real-time serving
- **Health checks** are configured for model availability
- **Scaling policies** are applied for load management

#### 3.3 Monitoring Configuration
**OpenShift Built-in Monitoring Stack:**

The product-recommender-system leverages **OpenShift's built-in monitoring infrastructure**:

**Prometheus/Thanos Configuration:**
- **OpenShift Monitoring**: Uses OpenShift's built-in Prometheus/Thanos stack
- **Metrics Collection**: Automatically collects metrics from all deployed components
- **Service Discovery**: OpenShift automatically discovers and scrapes metrics endpoints
- **Long-term Storage**: Thanos provides long-term metrics retention

**Grafana Configuration:**
- **OpenShift Grafana**: Uses OpenShift's built-in Grafana instance
- **Dashboard Access**: Available through OpenShift Console → Monitoring → Dashboards
- **Pre-built Dashboards**: OpenShift provides default dashboards for Kubernetes metrics
- **Custom Dashboards**: Can be created for application-specific metrics

**Metrics Endpoints:**
```yaml
# Backend metrics endpoint (automatically discovered by OpenShift)
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/path: "/metrics"
  prometheus.io/port: "8000"
```

**Monitoring Access:**
- **OpenShift Console**: Navigate to Monitoring → Dashboards
- **Grafana URL**: Available through OpenShift Console
- **Prometheus URL**: Available through OpenShift Console
- **Metrics API**: Direct access to Prometheus/Thanos API

**What's Monitored:**
- **Application metrics**: Request rates, response times, error rates
- **Infrastructure metrics**: CPU, memory, disk usage
- **Database metrics**: PostgreSQL performance and connections
- **Model metrics**: Training times, inference latency, model accuracy
- **Feast metrics**: Feature store performance and serving latency

### Phase 4: User Interaction Flow

#### 4.1 New User Onboarding
```
User visits application → Selects preferences → Gets personalized recommendations
```

**What happens:**
- **User registration** with basic profile information
- **Preference selection** from available categories
- **User embedding** is generated from preferences
- **Initial recommendations** are served based on preferences

#### 4.2 Existing User Experience
```
User logs in → System loads preferences → Serves personalized recommendations
```

**What happens:**
- **User authentication** with JWT tokens
- **Preference retrieval** from user profile
- **Recommendation serving** from cached results
- **Real-time updates** based on recent interactions

#### 4.3 Product Discovery & Search

**Text-Based Search:**
```
User enters search query → Semantic embedding → Vector similarity search → Results
```

**Image-Based Search:**
```
User uploads image → CLIP encoding → Vector similarity search → Similar products
```

**What happens:**
- **Query embedding** using the same model as training
- **Vector similarity search** in pgvector database
- **Result ranking** by similarity score
- **Product display** with images and details

#### 4.4 Shopping Cart Integration

**Adding Items to Cart:**
```
User clicks "Add to Cart" → Backend validates → Cart updated → UI refreshed
```

**What happens:**
- **Item validation** (availability, price, etc.)
- **Cart state management** in backend database
- **Real-time updates** to frontend
- **Persistent storage** across sessions

**Cart Management:**
```
User views cart → Modifies quantities → Proceeds to checkout → Purchase completed
```

**What happens:**
- **Cart retrieval** from user session
- **Quantity updates** with validation
- **Checkout process** with payment integration
- **Purchase recording** in interaction history

### Phase 5: Continuous Learning & Updates

#### 5.1 Interaction Tracking
```python
# Record user interactions for model updates
def record_interaction(user_id, item_id, interaction_type):
    interaction = {
        "user_id": user_id,
        "item_id": item_id,
        "timestamp": datetime.now(),
        "interaction_type": interaction_type  # view, cart, purchase, rate
    }
    store_interaction(interaction)
```

**What happens:**
- **User actions** are tracked (views, cart additions, purchases)
- **Interaction data** is stored in feature store
- **Real-time updates** to user preferences
- **Model retraining** triggers based on new data

#### 5.2 Model Retraining
```bash
# Automated retraining pipeline
kubectl create job --from=cronjob/model-retraining
```

**What happens:**
- **New interaction data** is loaded from feature store
- **Model retraining** with updated datasets
- **Performance evaluation** against previous models
- **Model deployment** if performance improves

**Retraining Schedule:**
```yaml
# From run-pipeline-job.yaml
schedule: "0 0 * * *"  # Run daily at midnight
concurrencyPolicy: Forbid
```

**Schedule Details:**
- **Frequency**: **Daily at midnight (00:00 UTC)**
- **Cron Expression**: `"0 0 * * *"` (minute hour day month weekday)
- **Concurrency Policy**: `Forbid` (prevents overlapping runs)
- **Timezone**: UTC

**What this means:**
- **Daily retraining** - Models are retrained every day at midnight
- **No overlapping runs** - Only one training job runs at a time (prevents multiple concurrent training jobs)
- **Automatic updates** - New interaction data is incorporated daily
- **Performance monitoring** - Models are evaluated against previous versions

**What Happens During Midnight Training:**

**Phase 1: Data Loading & Preparation**
```python
# Load data from Feast feature store
load_data_task = load_data_from_feast()
item_df = pd.read_parquet(item_df_input.path)
user_df = pd.read_parquet(user_df_input.path)
interaction_df = pd.read_parquet(interaction_df_input.path)
```
- **Load user data** - User profiles and preferences from Feast
- **Load item data** - Product catalog and features from Feast
- **Load interaction data** - User-item interactions from Feast
- **Data validation** - Ensure data quality and completeness

**Phase 2: Model Training**
```python
# Train two-tower neural network
item_encoder, user_encoder, models_definition = create_and_train_two_tower(
    item_df, user_df, interaction_df, return_model_definition=True
)
```
- **UserTower training** - Neural network for user embeddings
- **ItemTower training** - Neural network for item embeddings
- **Two-tower architecture** - Collaborative filtering with deep learning
- **Model optimization** - Loss minimization and gradient descent

**Training Data Sources:**

**1. Static Dataset (Primary)**
```python
# Load from Feast feature store
dataset_provider = LocalDatasetProvider(store)
item_df = dataset_provider.item_df()           # recommendation_items.parquet
user_df = dataset_provider.user_df()           # recommendation_users.parquet
interaction_df = dataset_provider.interaction_df()  # recommendation_interactions.parquet
```
- **User data**: User profiles, preferences, demographics
- **Item data**: Product catalog, features, categories, pricing
- **Interaction data**: User-item interactions (views, cart, purchase, ratings)

**2. Streaming Data (Real-time)**
```python
# Additional data from PostgreSQL database
if table_exists(engine, "new_users"):
    stream_users_df = pd.read_sql("SELECT * FROM new_users", engine)
    user_df = pd.concat([user_df, stream_users_df], axis=0)

if table_exists(engine, "stream_interaction"):
    stream_interaction_df = pd.read_sql("SELECT * FROM stream_interaction", engine)
    interaction_df = pd.concat([interaction_df, stream_interaction_df], axis=0)
```
- **New users**: Real-time user registrations and preferences
- **Stream interactions**: Live user interactions (views, purchases, ratings)
- **Data fusion**: Combines static and streaming data for training

**3. Data Processing Pipeline**
```python
# Data preprocessing and feature engineering
dataset = preproccess_pipeline(item_df, user_df, interaction_df)
```
- **Feature extraction**: Numerical, categorical, and text features
- **Text embedding**: BGE model for product descriptions
- **Interaction weighting**: Calculates interaction strength and magnitude
- **Data alignment**: Matches users, items, and interactions

**Training Data Structure:**
- **User features**: Demographics, preferences, signup date
- **Item features**: Categories, pricing, ratings, product descriptions
- **Interaction features**: Interaction type, timestamp, rating, quantity
- **Magnitude calculation**: Interaction strength based on type and frequency

**Phase 3: Embedding Generation**
```python
# Generate embeddings for all users and items
item_embed_df["embedding"] = item_encoder(**proccessed_items).detach().numpy().tolist()
user_embed_df["embedding"] = user_encoder(**proccessed_users).detach().numpy().tolist()
```
- **User embeddings** - Generate vector representations for all users
- **Item embeddings** - Generate vector representations for all items
- **Batch processing** - Efficient embedding generation for large datasets
- **Vector storage** - Store embeddings in PostgreSQL with PGVector

**Phase 4: Model Registration & Deployment**
```python
# Save trained models
torch.save(item_encoder.state_dict(), item_output_model.path)
torch.save(user_encoder.state_dict(), user_output_model.path)

# Register in Model Registry
create_model_registry_task = registry_model_to_model_registry(
    author=fetch_api_credentials_task.outputs["author"],
    user_token=fetch_api_credentials_task.outputs["user_token"],
    host=fetch_api_credentials_task.outputs["host"],
    bucket_name=train_model_task.outputs["bucket_name"],
    new_version=train_model_task.outputs["new_version"],
    object_name=train_model_task.outputs["object_name"],
    torch_version=train_model_task.outputs["torch_version"],
)
```
- **Model serialization** - Save trained models to disk
- **Version management** - Increment model version numbers
- **Model registry** - Register new models in central registry
- **Deployment preparation** - Prepare models for production serving

**Where Training Knowledge is Stored:**

**1. Model Artifacts (MinIO Object Storage)**
```python
# Store trained models in MinIO
minio_client.fput_object(
    bucket_name="user-encoder",
    object_name=f"user-encoder-{new_version}.pth",
    file_path=user_output_model.path,
)
```
- **UserTower model**: `user-encoder-{version}.pth` in MinIO bucket
- **ItemTower model**: `item-encoder-{version}.pth` in MinIO bucket
- **Model configurations**: JSON files with model architecture details
- **Version tracking**: Incremental versioning (1.0.0, 1.0.1, etc.)

**2. Model Registry (Centralized Management)**
```python
# Register models in Model Registry
registry_model_to_model_registry(
    bucket_name=bucket_name,
    new_version=new_version,
    object_name=object_name,
    torch_version=torch_version,
)
```
- **Model metadata**: Version, architecture, performance metrics
- **Model lineage**: Training history and data sources
- **Deployment tracking**: Which models are in production
- **A/B testing**: Model comparison and rollback capabilities

**3. Feature Store (PostgreSQL + PGVector)**
```python
# Push embeddings to online feature store
store.push(
    "item_embed_push_source",
    item_embed_df,
    to=PushMode.ONLINE,
    allow_registry_cache=False,
)
```
- **User embeddings**: Vector representations of all users
- **Item embeddings**: Vector representations of all items
- **Real-time serving**: Embeddings available for instant recommendations
- **Vector database**: PostgreSQL with PGVector extension

**4. Database Version Tracking (PostgreSQL)**
```python
# Track model versions in database
CREATE TABLE model_version (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
- **Version history**: Complete model version timeline
- **Deployment tracking**: Which version is currently active
- **Rollback capability**: Easy reversion to previous models
- **Audit trail**: Full training and deployment history

**5. Model Definition Files (JSON)**
```python
# Save model architecture definitions
models_definition = {
    "items_num_numerical": dataset.items_num_numerical,
    "items_num_categorical": dataset.items_num_categorical,
    "users_num_numerical": dataset.users_num_numerical,
    "users_num_categorical": dataset.users_num_categorical,
}
```
- **Architecture specs**: Model structure and feature dimensions
- **Feature mappings**: How input features map to model layers
- **Reproducibility**: Exact model configuration for rebuilding
- **Deployment config**: Model loading and serving parameters

**Phase 5: Feature Store Updates**
```python
# Push embeddings to online feature store
store.push(
    "item_embed_push_source",
    item_embed_df,
    to=PushMode.ONLINE,
    allow_registry_cache=False,
)
```
- **Embedding deployment** - Push new embeddings to online store
- **Real-time serving** - Update feature store for immediate use
- **Vector database** - Update PostgreSQL+PGVector with new embeddings
- **Cache invalidation** - Clear old embeddings and load new ones

**Training Job Details:**
- **Single comprehensive job** - One training pipeline that handles all model components
- **Complete retraining** - UserTower, ItemTower, and embeddings are all retrained together
- **Sequential processing** - Data loading → Model training → Embedding generation → Model registration
- **Single deployment** - All updated models are deployed as a complete set

**Why "Only one training job runs at a time":**
- **Resource efficiency** - Prevents multiple GPU-intensive training jobs from competing
- **Data consistency** - Ensures all models are trained on the same dataset snapshot
- **Deployment safety** - Prevents conflicts when registering new model versions
- **Monitoring clarity** - Single training run is easier to track and debug

### Data Flow Summary

```
Data Generation → Feature Store → Model Training → Model Registry →
Deployment → User Interaction → Interaction Tracking → Model Retraining
```

### Key Integration Points

1. **Feast Feature Store**: Central data hub for all features
2. **Model Registry**: Version control for ML models
3. **pgvector Database**: Vector similarity search
4. **FastAPI Backend**: Real-time recommendation serving
5. **React Frontend**: User interface and interactions
6. **Kubeflow Pipelines**: Automated training workflows

This workflow ensures a complete, production-ready recommendation system that continuously learns and improves based on user interactions.

## 📋 Architecture Diagrams

### Data Processing Pipeline
<img src="figures/data_processing_pipeline.drawio.png" alt="Data Processing Pipeline" width="80%">

### Training & Batch Scoring

#### Recommendation Algorithm Stages:

1. **Filtering**
   Removes invalid candidates based on user demographics (e.g., age, item availability in the region) and previously viewed items.

2. **Ranking**
   Identifies the most relevant top-k items based on previous interactions between users and items (trained with two-tower algorithm).

3. **Business Ordering**
   Reorders candidates according to business logic and priorities.

#### Training Process
* Feast takes the Raw data (item table, user table, interaction table) and stores the items, users, and interactions as Feature Views.
* Using the Two-Tower architecture technique, we train the item and user encoders based on the existing user-item interactions.

<img src="figures/training_and_batch_scoring.drawio.png" alt="Training & Batch scoring" width="80%">

#### Batch Scoring
* After completing the training of the Encoders, embed all items and users, then push them in the PGVector database as embedding.
* Because we use batch scoring, we calculate for each user the top k recommended items using the item embeddings
* Pushes this top k items for each user to the online store Feature Store.

### Inference

#### Existing User Case:
* Sending a get request from the EDB vectorDB to get the embedding of the existing user.
* Perform a similarity search on the item vectorDB to get the top k similar items.

#### New User Case:
* The new users will be embedded into a vector representation.
* The user vector will do a similarity search from the EDB PGVector to get the top k suggested items

<img src="figures/Inference.drawio.png" alt="Inference" width="80%">

### Search by Text & Search by Image
1. Embed the user query into embeddings.
2. Search the top-k closest items that were generated with the same model at batch inference time.
3. Return to user the recommended items

<img src="figures/search_by.drawio.png" alt="Search by Text/Image" width="80%">

## 🔍 How to Verify System Health

### System Health Verification

The product-recommendation-system provides multiple ways to verify it's working properly:

#### 1. **Health Check Endpoints**

**Backend Health Checks:**
```bash
# Liveness check - basic system availability
curl http://your-backend-url/health/live
# Expected: {"status": "alive"}

# Readiness check - full system readiness including database
curl http://your-backend-url/health/ready
# Expected: {"status": "ready"}
```

**Frontend Health Check:**
```bash
# Frontend availability
curl http://your-frontend-url/
# Expected: 200 OK with application content
```

**Feast Health Check:**
```bash
# Feature store health
curl http://your-feast-url/health
# Expected: 200 OK
```

#### 2. **Integration Tests**

**Run Automated Tests:**
```bash
# Run comprehensive integration tests
./run_integration_tests.sh tests/integration/test_endpoints.tavern.yaml
```

**Test Coverage:**
- ✅ **Frontend Health**: Verifies UI is accessible
- ✅ **Backend Health**: Checks API availability and database connectivity
- ✅ **Feast Health**: Validates feature store functionality
- ✅ **Response Validation**: Ensures correct JSON responses

#### 3. **Monitoring & Metrics**

**OpenShift Monitoring Dashboard:**
```bash
# Access monitoring through OpenShift Console
# Navigate to: Monitoring → Dashboards
```

**Key Metrics to Monitor:**
- ✅ **Application Metrics**: Request rates, response times, error rates
- ✅ **Infrastructure Metrics**: CPU, memory, disk usage
- ✅ **Database Metrics**: PostgreSQL performance and connections
- ✅ **Model Metrics**: Training times, inference latency, model accuracy
- ✅ **Feast Metrics**: Feature store performance and serving latency

**Metrics Endpoints:**
```yaml
# Backend metrics endpoint (automatically discovered by OpenShift)
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/path: "/metrics"
  prometheus.io/port: "8000"
```

#### 4. **Functional Verification**

**Recommendation API Test:**
```bash
# Test recommendation generation
curl -X POST http://your-backend-url/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user"}'
# Expected: List of recommended products
```

**Search Functionality Test:**
```bash
# Test text search
curl -X POST http://your-backend-url/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "smartphone"}'
# Expected: Relevant product search results
```

**User Management Test:**
```bash
# Test user creation and preferences
curl -X POST http://your-backend-url/users \
  -H "Content-Type: application/json" \
  -d '{"name": "test_user", "preferences": ["electronics"]}'
# Expected: User created successfully
```

#### 5. **Training Pipeline Verification**

**Check Training Job Status:**
```bash
# Verify daily training pipeline is running
oc get cronjob kfp-run-job -n your-namespace
# Expected: Active cronjob with last successful run
```

**Model Registry Check:**
```bash
# Verify models are being registered
oc get pods -l app=model-registry -n your-namespace
# Expected: Model registry pods running
```

**Feature Store Verification:**
```bash
# Check feature store data
oc get pods -l app=feast -n your-namespace
# Expected: Feast pods running with data
```

#### 6. **Performance Benchmarks**

**Response Time Verification:**
- ✅ **API Response**: < 500ms for recommendations
- ✅ **Search Response**: < 1s for text/image search
- ✅ **Database Queries**: < 100ms for feature retrieval
- ✅ **Model Inference**: < 200ms for embedding generation

**Throughput Verification:**
- ✅ **Concurrent Users**: System handles multiple simultaneous requests
- ✅ **Recommendation Quality**: Relevant and diverse product suggestions
- ✅ **Search Accuracy**: High precision and recall for search queries

#### 7. **Error Monitoring**

**Check for Errors:**
```bash
# View application logs
oc logs -f deployment/backend -n your-namespace

# Check for error patterns
oc logs deployment/backend -n your-namespace | grep ERROR
```

**Common Issues to Monitor:**
- ❌ **Database Connection Failures**: Check PostgreSQL connectivity
- ❌ **Model Loading Errors**: Verify model files in MinIO
- ❌ **Feature Store Errors**: Check Feast configuration
- ❌ **Training Pipeline Failures**: Monitor daily training jobs

#### 8. **End-to-End Testing**

**Complete User Journey:**
1. ✅ **User Registration**: Create new user account
2. ✅ **Preference Selection**: Set user preferences
3. ✅ **Recommendation Generation**: Get personalized recommendations
4. ✅ **Product Search**: Search by text or image
5. ✅ **Cart Management**: Add items to cart
6. ✅ **Checkout Process**: Complete purchase flow

### Success Criteria

**System is working properly when:**
- ✅ All health endpoints return 200 OK
- ✅ Integration tests pass (100% success rate)
- ✅ Response times meet performance benchmarks
- ✅ Training pipeline runs successfully daily
- ✅ Models are being updated and registered
- ✅ Feature store contains current data
- ✅ User interactions generate relevant recommendations
- ✅ Search functionality returns accurate results
- ✅ No critical errors in application logs
- ✅ Monitoring dashboards show healthy metrics

## 🔧 Requirements

Depend on the scale and speed required, for small amount of users have minimum of:
* No GPU required; for larger scale and faster performance, use GPUs.
* 4 CPU cores.
* 16 Gi of RAM.
* Storage: 8 Gi (depend on the input dataset).

### Required Software

* `oc` command installed
* `helm` command installed
* Red Hat OpenShift.
* Red Hat OpenShift AI version 2.2 and above.
* Red Hat Authorino Operator (stable update channel, version 1.2.1 or later)
* Red Hat OpenShift Serverless Operator
* Red Hat OpenShift Service Mesh Operator

#### Make sure you have configured
Under openshiftAI DataScienceCluster CR change modelregistry, and feastoperator to `Managed` state which by default are on `Removed`:
```yaml
apiVersion: datasciencecluster.opendatahub.io/v1
kind: DataScienceCluster
metadata:
  name: default-dsc
...
spec:
  components:
    codeflare:
      managementState: Managed
    kserve:
      managementState: Managed
      nim:
        managementState: Managed
      rawDeploymentServiceConfig: Headless
      serving:
        ingressGateway:
          certificate:
            secretName: rhoai-letscrypt-cert
            type: Provided
        managementState: Managed
        name: knative-serving
    modelregistry:
      managementState: Managed
      registriesNamespace: rhoai-model-registries
    feastoperator:
      managementState: Managed
    trustyai:
      managementState: Managed
    kueue:
      managementState: Managed
    workbenches:
      managementState: Managed
      workbenchNamespace: rhods-notebooks
    dashboard:
      managementState: Managed
    modelmeshserving:
      managementState: Managed
    datasciencepipelines:
      managementState: Managed
```

### Required Permissions

* Standard user. No elevated cluster permissions required

## 🚀 Installation

### Quick Start

1. Fork and clone the repository:
   ```bash
   # Fork via GitHub UI, then:
   git clone https://github.com/<your-username>/product-recommender-system.git
   cd product-recommender-system
   ```

2. Navigate to the helm directory:
   ```bash
   cd helm/
   ```

3. Set the namespace environment variable to define on which namespace the quickstart will be installed:
   ```bash
   # Replace <namespace> with your desired namespace
   export NAMESPACE=<namespace>
   ```

4. Install using make (this should take 8~ minutes with the default data, and with custom data maybe less or more):
   ```bash
   # This will create the namespace and deploy all components
   make install
   ```

* Or installing and defining a namespace together:
   ```bash
   # Replace <namespace> with your desired namespace and install in one command
   make install NAMESPACE=<namespace>
   ```

### Custom Dataset Configuration

By default, a dataset is automatically generated when the application is installed on the cluster.

To use a custom dataset instead, provide a URL by setting the `DATASET_URL` property during installation:

```bash
# Replace <custom_dataset_url> with the desired dataset URL
make install DATASET_URL=<custom_dataset_url>
```

### Advanced Configuration

For detailed configuration options, see the [helm/README.md](helm/) for deployment options and [backend/README.md](backend/) for API configuration.

## 🧹 Uninstallation

To uninstall the recommender system and clean up resources:

1. Navigate to the helm directory:
   ```bash
   cd helm/
   ```

2. Uninstalling with namespace specified:
   ```bash
   # Replace <namespace> with your namespace
   make uninstall NAMESPACE=<namespace>
   ```

## 📚 Documentation

### Component-Specific Documentation

- **[recommendation-core/README.md](recommendation-core/README.md)** - ML library documentation, model architectures, and data generation
- **[recommendation-training/README.md](recommendation-training/README.md)** - Training pipeline documentation, Kubeflow workflows, and model registration
- **[recommendation-model-registry/README.md](recommendation-model-registry/README.md)** - Infrastructure tools documentation and cluster credential management
- **[backend/README.md](backend/)** - API documentation, authentication, and service configuration
- **[frontend/README.md](frontend/)** - User interface documentation and component architecture
- **[helm/README.md](helm/)** - Deployment documentation, Helm charts, and Kubernetes configuration

### Development Guides

- **Model Development**: See [recommendation-core/README.md](recommendation-core/README.md) for adding new model architectures
- **Training Pipeline**: See [recommendation-training/README.md](recommendation-training/README.md) for modifying training workflows
- **API Development**: See [backend/README.md](backend/) for extending the REST API
- **UI Development**: See [frontend/README.md](frontend/) for frontend component development

### Troubleshooting

- **Deployment Issues**: Check [helm/README.md](helm/) for common deployment problems
- **Training Problems**: See [recommendation-training/README.md](recommendation-training/README.md) for pipeline troubleshooting
- **Model Issues**: Refer to [recommendation-core/README.md](recommendation-core/README.md) for model debugging
- **Infrastructure**: Check [recommendation-model-registry/README.md](recommendation-model-registry/README.md) for cluster access issues

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch** for your changes
3. **Make changes** in the appropriate component directory
4. **Update documentation** in the relevant README files
5. **Test your changes** using the component-specific testing procedures
6. **Submit a pull request** with detailed description

### Component-Specific Guidelines

- **recommendation-core**: Follow the development guidelines in [recommendation-core/README.md](recommendation-core/README.md)
- **recommendation-training**: See contribution guidelines in [recommendation-training/README.md](recommendation-training/README.md)
- **backend**: Follow API development guidelines in [backend/README.md](backend/)
- **frontend**: See UI development guidelines in [frontend/README.md](frontend/)

## 📄 License

This project is licensed under the same terms as the Red Hat AI Quickstart program.

## 🔗 References

- [Red Hat OpenShift AI Documentation](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai)
- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [Feast Documentation](https://docs.feast.dev/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [React Documentation](https://react.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Note**: This is a comprehensive recommendation system designed for production use. Always test thoroughly in development environments before deploying to production. For component-specific questions, refer to the individual README files in each subdirectory.
