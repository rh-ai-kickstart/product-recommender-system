# Installation Guide

Complete instructions for deploying the Product Recommender System in production environments.

## Prerequisites

### Hardware Requirements

**Minimum (small-scale deployment):**
- 4 CPU cores
- 16 GB RAM
- 8 GB storage (varies with dataset size)
- No GPU required (recommended for larger scale)

**Recommended (production):**
- 8+ CPU cores
- 32+ GB RAM
- 50+ GB storage
- GPU for faster ML inference (optional)

### Software Prerequisites

- **OpenShift CLI** (`oc`) installed and configured
- **Helm 3.x** for Kubernetes deployments
- **Red Hat OpenShift** cluster access
- **Red Hat OpenShift AI** (version 2.2+)
- **Red Hat Authorino Operator** (stable channel, v1.2.1+)
- **Red Hat OpenShift Serverless** and **Service Mesh** Operators

### Required Permissions

Standard user access - no elevated cluster permissions required.

## Production Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/product-recommender-system.git
cd product-recommender-system
```

### 2. Deploy to OpenShift

**Basic installation:**
```bash
cd helm/
make install NAMESPACE=recommender-system
```

**With custom dataset:**
```bash
make install NAMESPACE=recommender-system DATASET_URL=<your-dataset-url>
```

**With additional Helm arguments:**
```bash
make install NAMESPACE=recommender-system EXTRA_HELM_ARGS="--set key=value"
```

Installation typically takes 8-10 minutes with the default dataset.

### 3. Verify Installation

Check deployment status:
```bash
make status NAMESPACE=recommender-system
```

This will show:
- Running pods
- Active services  
- Available routes
- Persistent Volume Claims (PVCs)

### 4. Access Your Application

After successful installation, the application will be available through OpenShift routes. Use the status command above to find the frontend URL.

## Custom Dataset Integration

### Dataset Format Requirements

Your custom dataset should include:
- **Users table** - User demographics and preferences
- **Items table** - Product information and features
- **Interactions table** - User-item interaction history

### Supported Formats

- Parquet files (recommended)
- CSV files
- JSON format

### Integration Process

1. **Prepare your dataset** following the expected schema
2. **Upload to accessible location** (S3, HTTP endpoint, etc.)
3. **Deploy with dataset URL:**
   ```bash
   make install NAMESPACE=recommender-system DATASET_URL=<your-dataset-url>
   ```

The system will automatically process and integrate your data during deployment.

## Configuration

### Helm Chart Values

Key configuration options in `helm/product-recommender-system/values.yaml`:

```yaml
# Application configuration
app:
  namespace: recommender-system
  datasetUrl: ""  # Custom dataset URL

# Resource limits
resources:
  backend:
    requests:
      memory: "2Gi"
      cpu: "1"
    limits:
      memory: "4Gi" 
      cpu: "2"
```

## Troubleshooting

### Common Issues

**Installation fails with timeout:**
- Increase timeout: `make install NAMESPACE=<ns> EXTRA_HELM_ARGS="--timeout=20m"`
- Check cluster resources: `oc get nodes` and `oc describe nodes`

**Pods stuck in pending state:**
- Check resource availability: `oc describe pod <pod-name>`
- Verify PVC creation: `oc get pvc`

**Application not accessible:**
- Check routes: `oc get routes`
- Verify services: `oc get svc`
- Check pod logs: `oc logs <pod-name>`

### Log Access

**Backend logs:**
```bash
oc logs deployment/backend -f
```

**Frontend logs:**
```bash
oc logs deployment/frontend -f
```

**ML pipeline logs:**
```bash
oc get jobs
oc logs job/<job-name>
```

## Uninstallation

To remove the recommender system:

```bash
cd helm/
make uninstall NAMESPACE=recommender-system
```

**Note:** To completely remove the namespace:
```bash
oc delete project recommender-system
```

## Next Steps

After successful installation:

1. **Explore the application** using the provided routes
2. **Review the API documentation** at `<backend-route>/docs`
3. **Check the admin dashboard** for system monitoring
4. **Import your data** if using custom datasets
5. **Configure monitoring** and alerts for production use

For development and customization, see [CONTRIBUTING.md](CONTRIBUTING.md).