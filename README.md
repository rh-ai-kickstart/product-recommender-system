# AI Kickstart - Product Recommender System

Welcome to the Product Recommender System Kickstart!
Use this to quickly get a recommendation engine with user-item relationships up and running in your environment.

To see how it's done, jump straight to [installation](#install).

## 🎯 Description
The Product Recommender System Kickstart enables the rapid establishment of a scalable and personalized product recommendation service.

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

#### [recommendation-model-registry/](recommendation-model-registry/README.md)
**Model Registry Infrastructure** - Centralized model management and deployment system
- **Model versioning**: Comprehensive model lifecycle management with version control
- **Model serving**: RESTful API endpoints for model inference and prediction serving
- **Model metadata**: Detailed tracking of model artifacts, parameters, and performance metrics
- **Deployment automation**: Automated model deployment to production environments
- **Model monitoring**: Real-time model performance monitoring and alerting
- **A/B testing**: Support for model comparison and gradual rollout strategies
- **Security**: Model access control, authentication, and audit logging

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
- **Monitoring**: Prometheus and Grafana integration for comprehensive observability
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

## 🔧 Requirements

### Minimum Hardware Requirements

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

3. Set the namespace environment variable to define on which namespace the kickstart will be installed:
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

This project is licensed under the same terms as the Red Hat AI Kickstart program.

## 🔗 References

- [Red Hat OpenShift AI Documentation](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai)
- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [Feast Documentation](https://docs.feast.dev/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [React Documentation](https://react.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Note**: This is a comprehensive recommendation system designed for production use. Always test thoroughly in development environments before deploying to production. For component-specific questions, refer to the individual README files in each subdirectory.

