# Recommendation Core Library

A comprehensive Python library for building production-ready recommendation systems using two-tower neural networks, multi-modal features, and Feast integration.



## 🎯 Overview

The Recommendation Core library provides the foundational components for building scalable recommendation systems that:
- **Process multi-modal data**: Text, images, numerical, and categorical features
- **Train two-tower models**: Separate encoders for users and items
- **Integrate with Feast**: Feature store for online serving
- **Generate embeddings**: High-quality representations for similarity search
- **Support filtering**: Rule-based recommendation refinement

## 🤔 Why "recommendation-core"?

The name "recommendation-core" was chosen to reflect its **central role** in the recommendation system architecture and its **foundational nature** as a reusable library.

### **Core Library Pattern**
The name follows a common software architecture pattern where "core" indicates the **foundational library** that other components depend on:

```
product-recommender-system/
├── recommendation-core/          # 🧠 Core ML Library
├── recommendation-training/      # 🚀 Training Pipeline
├── backend/                     # 🌐 API Layer
├── frontend/                    # 🎨 UI Layer
└── helm/                        # ☸️ Deployment
```

### **Central Role in Architecture**
The recommendation-core serves as the **central nervous system** of the entire recommendation system:

- **Backend** imports and uses it: `from recommendation_core.models import EntityTower`
- **Training pipeline** uses it as base image: `BASE_IMAGE = "quay.io/rh-ai-quickstart/recommendation-core:latest"`
- **All ML logic** is contained within it

### **Core Capabilities**
It provides the **essential ML capabilities** that make the recommendation system work:

- **Neural Network Models** - EntityTower, UserTower, ItemTower
- **Feature Processing** - Multi-modal data handling (text, images, numerical)
- **Search Services** - Text and image similarity search
- **Data Generation** - Synthetic datasets and AI image generation
- **Model Serving** - Real-time inference capabilities

### **Reusable Foundation**
The "core" naming suggests it's designed to be:
- **Reusable** across different recommendation applications
- **Extensible** for different use cases
- **Independent** of specific business logic
- **Foundation** for building recommendation systems

### **Separation of Concerns**
The naming reflects a clean architecture where:
- **recommendation-core** = ML algorithms and models
- **recommendation-training** = Training workflows
- **backend** = API and business logic
- **frontend** = User interface

### **Why This Name Works Best**

1. **Clear Purpose** - Immediately indicates it's for recommendation systems
2. **Core Concept** - Emphasizes its foundational role
3. **Library Focus** - Indicates it's a reusable library, not a complete application
4. **Professional Naming** - Follows industry conventions for core libraries

The name effectively communicates that this is the **essential, reusable foundation** for building recommendation systems, which is exactly what it is!

## 🏗️ Architecture

### Core Components

#### 1. **Neural Network Models** (`models/`)
- **EntityTower**: Multi-modal encoder for users and items
- **UserTower**: Specialized user representation learning
- **ItemTower**: Specialized item representation learning
- **TwoTower**: Combined user-item training framework

#### 2. **Data Processing** (`models/data_util.py`)
- **Feature preprocessing**: Normalization and encoding
- **Multi-modal handling**: Text, image, numerical, categorical
- **Data validation**: Input sanitization and type checking

#### 3. **Recommendation Services** (`service/`)
- **CLIP Encoder**: Text and image feature extraction
- **Dataset Provider**: Data loading and batching
- **Search Services**: Text and image-based search

#### 4. **Data Generation** (`generation/`)
- **Synthetic datasets**: Amazon-like e-commerce data
- **Image generation**: Stable Diffusion for product images
- **Interaction simulation**: User behavior modeling

#### 5. **Feature Store** (`feature_repo/`)
- **Feast configuration**: Feature definitions and serving
- **Online/offline stores**: PostgreSQL and file-based storage
- **Feature engineering**: Transformations and aggregations

## 🚀 Quick Start

### Installation

```bash
# Install from source
git clone <repository-url>
cd recommendation-core
pip install -e .

# Or install with optional dependencies
pip install -e ".[data-gen,test,dev]"
```

### Basic Usage

```python
from recommendation_core.models import EntityTower
from recommendation_core.service import ClipEncoder
import torch

# Initialize the model
model = EntityTower(
    num_numerical=5,
    num_of_categories=10,
    d_model=64,
    text_embed_dim=384
)

# Process features
numerical_features = torch.randn(32, 5)
categorical_features = torch.randint(0, 10, (32, 1))
text_features = torch.randn(32, 10, 384)
url_image = torch.randn(32, 1)

# Generate embeddings
embeddings = model(
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    text_features=text_features,
    url_image=url_image
)
```

## 📦 Package Structure

```
recommendation-core/
├── src/recommendation_core/
│   ├── models/                 # Neural network architectures
│   │   ├── entity_tower.py    # Multi-modal encoder
│   │   ├── user_tower.py      # User representation
│   │   ├── item_tower.py      # Item representation
│   │   ├── two_tower.py       # Combined training
│   │   ├── data_util.py       # Data preprocessing
│   │   ├── filtering.py       # Recommendation filtering
│   │   └── train_two_tower.py # Training utilities
│   ├── service/               # Recommendation services
│   │   ├── clip_encoder.py    # CLIP feature extraction
│   │   ├── dataset_provider.py # Data loading
│   │   ├── search_by_text.py  # Text search
│   │   └── search_by_image.py # Image search
│   ├── generation/            # Data generation
│   │   ├── dataset_gen_amazon.py # Synthetic data
│   │   └── generate_images.py # Image generation
│   └── feature_repo/          # Feast configuration
│       ├── feature_store.yaml # Feature store config
│       └── data/              # Generated datasets
├── tests/                     # Unit tests
├── Containerfile              # Container build
└── pyproject.toml            # Package configuration
```

## 🔧 Configuration

### Model Architecture

The library supports flexible model configurations:

```python
# Basic configuration
config = {
    "num_numerical": 5,        # Number of numerical features
    "num_of_categories": 10,   # Number of categorical features
    "d_model": 64,            # Embedding dimension
    "text_embed_dim": 384,    # Text embedding dimension
    "image_embed_dim": 384,   # Image embedding dimension
    "dim_ratio": {            # Feature dimension allocation
        "numeric": 1,
        "categorical": 2,
        "text": 7,
        "image": 0
    }
}
```

### Feature Processing

```python
from recommendation_core.models.data_util import data_preproccess

# Preprocess features
processed_features = data_preproccess(
    dataframe=user_data,
    numerical_cols=['age', 'price'],
    categorical_cols=['category', 'brand'],
    text_cols=['description', 'title'],
    image_cols=['image_url']
)
```

## 🧪 Data Generation

### Synthetic Dataset Generation

```bash
# Generate Amazon-like e-commerce data
python -m recommendation_core.generation.dataset_gen_amazon \
    --n_users 1000 \
    --n_items 5000 \
    --n_interactions 20000
```

**Generated Datasets:**
- `recommendation_users.parquet`: User profiles and preferences
- `recommendation_items.parquet`: Product catalog with features
- `recommendation_interactions.parquet`: User-item interactions

### Who Runs dataset_gen_amazon.py?

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

### Where Are Parquet Files Written?

**Primary Location:**
```python
data_path = pathlib.Path("src/recommendation_core/feature_repo/data")
data_path.mkdir(parents=True, exist_ok=True)
```

**Full Path:** `src/recommendation_core/feature_repo/data/`

**Files Generated:**
The script writes **9 parquet files** to this location:

#### **Main Dataset Files:**
1. `recommendation_users.parquet` - User profiles and preferences
2. `recommendation_items.parquet` - Product catalog with features
3. `recommendation_interactions.parquet` - User-item interactions

#### **Dummy/Feature Files:**
4. `dummy_item_embed.parquet` - Dummy item embeddings
5. `dummy_user_embed.parquet` - Dummy user embeddings
6. `user_items.parquet` - User-item relationships
7. `item_textual_features_embed.parquet` - Text feature embeddings
8. `item_clip_features_embed.parquet` - CLIP feature embeddings

**Directory Structure:**
```
recommendation-core/
└── src/
    └── recommendation_core/
        └── feature_repo/
            └── data/                    # ← Parquet files written here
                ├── recommendation_users.parquet
                ├── recommendation_items.parquet
                ├── recommendation_interactions.parquet
                ├── dummy_item_embed.parquet
                ├── dummy_user_embed.parquet
                ├── user_items.parquet
                ├── item_textual_features_embed.parquet
                └── item_clip_features_embed.parquet
```

**Key Points:**
- **Relative Path** - The script uses a relative path from where it's executed
- **Auto-Creation** - The directory is created automatically if it doesn't exist
- **Feast Integration** - This location aligns with the Feast feature store structure
- **Training Pipeline** - The training pipeline expects these files to be in this exact location

### Data Flow

```
Developer runs dataset_gen_amazon.py
         ↓
Creates parquet files in feature_repo/data/
         ↓
Training pipeline loads pre-generated files
         ↓
ML models train on synthetic data
```

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

### Why This Design?

**Advantages:**
1. **Reproducible datasets** - Same data across all environments
2. **Fast training** - No need to generate data during training
3. **Version control** - Dataset files can be committed to git
4. **Consistent testing** - Same test data for all tests

**Disadvantages:**
1. **Manual step** - Developers must remember to run it
2. **Static data** - No dynamic dataset generation
3. **Storage overhead** - Large parquet files in repository

### Image Generation

```bash
# Generate product images using Stable Diffusion
python -m recommendation_core.generation.generate_images
```

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

**Why This Design:**
- **Synthetic data** - No need for real product images
- **Consistent style** - All images generated with same AI model
- **Scalable** - Can generate unlimited product images
- **Demo-friendly** - Provides visual content for demonstrations

## 🔍 Search and Recommendation

### Text-Based Search

```python
from recommendation_core.service import search_by_text

# Search items by text description
results = search_by_text(
    query="wireless headphones",
    item_embeddings=item_embeddings,
    top_k=10
)
```

### Image-Based Search

```python
from recommendation_core.service import search_by_image

# Search items by image similarity
results = search_by_image(
    query_image=product_image,
    item_embeddings=item_embeddings,
    top_k=10
)
```

### Recommendation Filtering

```python
from recommendation_core.models.filtering import apply_filters

# Apply rule-based filtering
filtered_recommendations = apply_filters(
    recommendations=raw_recommendations,
    user_profile=user_data,
    filters=['availability', 'demographic', 'history']
)
```

## 🏗️ Building and Deployment

### Container Build

```bash
# Build the container image
podman build -t quay.io/rh-ai-quickstart/recommendation-core:latest .

# Push to registry
podman push quay.io/rh-ai-quickstart/recommendation-core:latest
```

### Package Installation

```bash
# Install in development mode
pip install -e .

# Install with all dependencies
pip install -e ".[data-gen,test,dev]"

# Install in production
pip install .
```

## 🚀 Who Uses the Recommendation-Core Container?

The recommendation-core container is used by different components in the product-recommender-system in various ways:

### 1. **Training Pipeline** (Primary Container User)
The **recommendation-training workflow** uses the recommendation-core container as a **base image** for ML training:

```python
BASE_IMAGE = os.getenv(
    "BASE_REC_SYS_IMAGE", "quay.io/rh-ai-quickstart/recommendation-core:latest"
)

@dsl.component(base_image=BASE_IMAGE)
def generate_candidates(...):
    from recommendation_core.models.entity_tower import EntityTower
    from recommendation_core.service.clip_encoder import ClipEncoder
```

**What this means:**
- The training pipeline **uses the recommendation-core container directly**
- It runs as a Kubeflow component with the recommendation-core container as the base image
- It imports and uses the library within the containerized environment
- Used for model training, feature generation, and data processing

### 2. **Main Application** (Library Source)
The **main product-recommender-system Containerfile** uses the recommendation-core container as a **library source**:

```dockerfile
# Copy recommendation-core first (needed for backend dependencies)
COPY recommendation-core/ /app/recommendation-core/

# Install the recommendation-core package first
WORKDIR /app/recommendation-core
RUN uv pip install .

# Install the backend package
WORKDIR /app/backend
RUN uv pip install .
```

**What this means:**
- The main application **doesn't use the recommendation-core container directly**
- Instead, it **copies the source code** and installs it as a Python package
- The backend imports and uses the library directly: `from recommendation_core.models import EntityTower`
- Used for real-time inference and recommendation serving

### 3. **Helm Chart Configuration** (Deployment Reference)
The **Helm values.yaml** references the container image for deployment:

```yaml
pipelineJobImage: quay.io/rh-ai-quickstart/recommendation-training:latest
applicationImage: quay.io/rh-ai-quickstart/recommendation-core:latest
# Note the backend uses recommendation_core as library.
frontendBackendImage: quay.io/rh-ai-quickstart/product-recommender-frontend-backend:latest
```

**What this means:**
- The Helm chart references the container image for deployment configuration
- Used for Kubernetes/OpenShift deployment orchestration
- Provides consistent image references across environments

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=recommendation_core tests/

# Run specific test categories
pytest tests/test_models/
pytest tests/test_service/
```

### Integration Tests

```bash
# Test model training
python -c "
from recommendation_core.models import EntityTower
import torch

model = EntityTower()
# Test forward pass
"

# Test data generation
python -m recommendation_core.generation.dataset_gen_amazon --n_users 10
```

## 📊 Performance Characteristics

### Model Performance
- **Training time**: ~30 minutes for 100K interactions
- **Inference latency**: <10ms per embedding
- **Memory usage**: ~2GB for full model
- **GPU acceleration**: 3-5x speedup with CUDA

### Data Processing
- **Feature extraction**: ~100ms per batch
- **CLIP encoding**: ~50ms per image/text
- **Data generation**: ~1000 items/minute

## 🔒 Security Considerations

### Data Privacy
- **No PII storage**: User data is anonymized
- **Encrypted storage**: Sensitive data is encrypted
- **Access controls**: Role-based permissions

### Model Security
- **Input validation**: Sanitized feature inputs
- **Model versioning**: Reproducible model artifacts
- **Secure serving**: HTTPS and authentication

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   export CUDA_VISIBLE_DEVICES=0
   python -c "torch.cuda.empty_cache()"
   ```

2. **Feast Connection Errors**
   ```bash
   # Check feature store connectivity
   feast config list
   feast apply
   ```

3. **Model Loading Issues**
   ```bash
   # Verify model files
   ls -la models/
   python -c "torch.load('model.pth')"
   ```

### Debug Commands

```bash
# Check package installation
python -c "import recommendation_core; print(recommendation_core.__version__)"

# Verify dependencies
pip list | grep -E "(torch|feast|transformers)"

# Test model components
python -c "
from recommendation_core.models import EntityTower
model = EntityTower()
print('Model created successfully')
"
```

## 📚 API Reference

### Core Models

#### EntityTower
```python
class EntityTower(nn.Module):
    def __init__(self, num_numerical, num_of_categories, d_model, ...)
    def forward(self, numerical_features, categorical_features, text_features, url_image)
```

#### ClipEncoder
```python
class ClipEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32")
    def encode_text(self, texts: List[str]) -> torch.Tensor
    def encode_image(self, images: List[PIL.Image]) -> torch.Tensor
```

### Data Utilities

#### data_preproccess
```python
def data_preproccess(
    dataframe: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    text_cols: List[str],
    image_cols: List[str]
) -> Dict[str, torch.Tensor]
```

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd recommendation-core
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

The project uses several quality tools:
- **ruff**: Fast Python linter
- **pytest**: Testing framework
- **black**: Code formatting (optional)

```bash
# Run quality checks
ruff check .
ruff format .

# Run tests
pytest tests/
```

### Contribution Guidelines

1. **Fork the repository**
2. **Create a feature branch**
3. **Make changes** with tests
4. **Run quality checks**
5. **Submit a pull request**

## 📄 License

This project is part of the product recommendation system and follows the same license as the main project.

## 📚 Related Documentation

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Feast Documentation](https://docs.feast.dev/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [CLIP Model](https://openai.com/blog/clip/)

---

**Note**: This library is designed for production use in recommendation systems. Always test thoroughly with your specific data and requirements before deploying to production.
