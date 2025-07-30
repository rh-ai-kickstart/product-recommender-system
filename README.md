# Product Recommender System Kickstart

A scalable, intelligent product recommendation platform with personalized user experiences, built for rapid deployment on Kubernetes/OpenShift.

## What is this?

This platform provides the tools to build and deploy an AI-powered product recommendation system that can:

- **Personalize recommendations** - Advanced ML algorithms using two-tower architecture for user-item matching
- **Handle new users** - Cold start recommendations using user preferences and demographics
- **Multi-modal search** - Text-based semantic search and image similarity search
- **Scale in production** - Kubernetes-ready architecture with microservices design
- **Real-time inference** - Fast recommendations using pre-computed embeddings and vector search

### Key Features

- 🎯 **Smart Recommendations** - Two-tower ML model with user-item embeddings and similarity search
- 🔍 **Advanced Search** - Semantic text search and visual similarity with image uploads
- 🛒 **Complete E-commerce** - Shopping cart, wishlist, orders, and user feedback system
- 🚀 **Production Ready** - Containerized deployment with Helm charts for Kubernetes/OpenShift
- ⚡ **Real-time Performance** - Vector database (pgvector) for fast similarity search and caching

## Quick Start

### Installation

**Option 1: Basic installation**
```bash
git clone https://github.com/<your-username>/product-recommender-system.git
cd product-recommender-system/helm
make install NAMESPACE=recommender-system
```

**Option 2: Custom dataset**
```bash
cd helm/
make install NAMESPACE=recommender-system DATASET_URL=<your-dataset-url>
```

**Access your app:**
- Frontend: Check your OpenShift routes after installation
- Backend API: `<backend-route>/docs` for API documentation

### Local Development

```bash
# Start backend
cd backend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && python init_backend.py
uvicorn main:app --reload &

# Start frontend (new React app)
cd ../frontend && npm install && npm run dev
```

📖 **[Full Installation Guide →](INSTALLING.md)**

## Project Structure

```
product-recommender-system/
├── frontend/           # React + TypeScript UI with modern components
├── backend/            # FastAPI server with ML recommendation engine
├── helm/               # Kubernetes deployment and Helm charts
├── figures/            # Architecture diagrams and documentation images
├── services/           # Feast feature store and Kafka services
└── ui/                 # Legacy UI (being phased out)
```

## Architecture Overview

The platform integrates several ML and data components:

- **React Frontend** - Modern TypeScript interface for product browsing and recommendations
- **FastAPI Backend** - ML-powered recommendation API with user management and e-commerce features
- **Two-Tower ML Model** - Advanced neural architecture for user-item embedding and similarity
- **Feast Feature Store** - Real-time and batch feature serving for ML models
- **pgvector Database** - Vector similarity search for fast recommendations and semantic search
- **Kafka Pipeline** - Real-time data processing for user interactions and feedback

### Data Processing Pipeline
<img src="figures/data_processing_pipeline.drawio.png" alt="Data Processing" width="80%">

### Training & Batch Scoring

#### Recommendation Algorithm Stages:

1. **Filtering** - Removes invalid candidates based on user demographics and availability
2. **Ranking** - Identifies top-k items using two-tower model trained on user-item interactions
3. **Business Ordering** - Reorders recommendations based on business logic and priorities

<img src="figures/training_and_batch_scoring.drawio.png" alt="Training & Batch scoring" width="80%">

### Real-time Inference

**Existing Users**: Fast lookup using pre-computed embeddings and similarity search
**New Users**: Real-time embedding generation with preference-based cold start

<img src="figures/Inference.drawio.png" alt="Inference" width="80%">

### Multi-modal Search
<img src="figures/search_by.drawio.png" alt="Search capabilities" width="80%">


## Getting Started Guides

### 👩‍💻 **For Developers**
- **[Contributing Guide](CONTRIBUTING.md)** - Development setup and workflow
- **[Backend API Reference](backend/README.md)** - Authentication, user management, and ML endpoints
- **[Frontend Development](frontend/README.md)** - React TypeScript setup and component architecture

### 🚀 **For Deployment**
- **[Installation Guide](INSTALLING.md)** - Production deployment on Kubernetes/OpenShift
- **[Helm Configuration](helm/README.md)** - Kubernetes deployment with Helm charts

### 🔧 **For ML Engineers**
- **[Feature Store Integration](backend/services/feast/)** - Feast setup for real-time and batch features
- **[Model Architecture](backend/README.md)** - Two-tower neural network implementation

## Example Use Cases

**E-commerce Recommendations**
```python
# Get personalized recommendations for existing user
GET /recommendations/{user_id}

# Real-time recommendations for new user
POST /recommendations
Authorization: Bearer {token}
{"num_recommendations": 10}
```

**Product Search**
```python
# Semantic text search
POST /products/search
{"query": "wireless headphones", "limit": 20}

# Image similarity search
POST /products/search/image
# Upload image file for similar product matching
```

## Production Installation

### Quick Installation

```bash
git clone https://github.com/<your-username>/product-recommender-system.git
cd product-recommender-system/helm
make install NAMESPACE=recommender-system
```

Installation typically takes 8-10 minutes with default dataset.

📖 **[Complete Installation Guide →](INSTALLING.md)**

## Development Commands

```bash
# Start everything locally
cd backend && python init_backend.py
uvicorn main:app --reload &
cd ../frontend && npm run dev

# Check production deployment status
cd helm && make status NAMESPACE=<your-namespace>

# Uninstall from cluster
cd helm && make uninstall NAMESPACE=<your-namespace>
```

## Community & Support

- **🐛 Issues** - [Report bugs and request features](https://github.com/<your-username>/product-recommender-system/issues)
- **💬 Discussions** - Ask questions about ML recommendations, deployment, or architecture
- **🤝 Contributing** - See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- **📚 Documentation** - Browse component READMEs and dedicated guides

## License

Built with ❤️ for rapid ML recommendation system deployment
