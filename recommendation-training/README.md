# Recommendation Training Container

This directory contains the container images for the recommendation training workflow using a two-image approach.

## Two-Image Architecture

### 1. Training Pipeline Image (`recommendation-training`)
- **Purpose**: Main ML training workflow
- **Base Image**: `registry.access.redhat.com/ubi9/python-311`
- **Contents**: Python dependencies, training scripts, ML libraries
- **Usage**: Training, data processing, model training

### 2. OC Tools Image (`recommendation-oc-tools`)
- **Purpose**: Cluster operations and model registry
- **Base Image**: `python:3.11-slim`
- **Contents**: OpenShift CLI, model registry tools
- **Usage**: `fetch_cluster_credentials()`, model registration

## Build Process

### Local Build - Training Image

```bash
cd recommendation-training
podman build -t quay.io/rh-ai-kickstart/recommendation-training:latest .
```

### Local Build - OC Tools Image

```bash
cd recommendation-training/oc-tools
podman build -t quay.io/rh-ai-kickstart/recommendation-oc-tools:latest .
```

### Push to Registry

```bash
# Push training image
podman push quay.io/rh-ai-kickstart/recommendation-training:latest

# Push oc-tools image
podman push quay.io/rh-ai-kickstart/recommendation-oc-tools:latest
```

### Automated Build

The containers are automatically built and pushed via GitHub Actions when:
- Changes are pushed to the `main` or `master` branch
- The workflows are manually triggered

**Workflows:**
- `.github/workflows/build-and-push.yaml` - Training pipeline image
- `.github/workflows/build-and-push-oc-tools.yaml` - OC tools image

## Container Contents

### Training Image (`recommendation-training`)
- **Base Image**: `registry.access.redhat.com/ubi9/python-311`
- **User**: `root`
- **Working Directory**: `/app`
- **Dependencies**: 
  - Python packages via `uv`
  - ML training libraries
  - Training scripts

### OC Tools Image (`recommendation-oc-tools`)
- **Base Image**: `python:3.11-slim`
- **Working Directory**: `/app`
- **Dependencies**: 
  - OpenShift CLI (`oc`)
  - `curl`, `tar`, `jq` utilities
  - `model_registry==0.2.21` Python package

## Key Files

### Training Image
- `Containerfile`: Container definition
- `train-workflow.py`: Kubeflow pipeline definition
- `entrypoint.sh`: Container entry point
- `pyproject.toml`: Python dependencies

### OC Tools Image
- `oc-tools/Containerfile`: Dedicated container for cluster operations
- Focused on OC CLI and model registry tools

## Usage

### Training Pipeline
```yaml
containers:
- name: kfp-runner
  image: quay.io/rh-ai-kickstart/recommendation-training:latest
  command: ['/bin/sh']
  args: ['-c', './entrypoint.sh']
```

### Cluster Operations
```python
@dsl.component(base_image="quay.io/rh-ai-kickstart/recommendation-oc-tools:latest")
def fetch_cluster_credentials():
    # OC CLI operations
```

## GitHub Actions

The automated build process requires:
- `QUAY_USERNAME`: Quay.io username
- `QUAY_PASSWORD`: Quay.io password/token

These secrets must be configured in the GitHub repository settings.

## Benefits of Two-Image Approach

1. **Separation of Concerns**: Training vs. cluster operations
2. **Smaller Images**: Each image focused on specific purpose
3. **Better Security**: Minimal tools in each image
4. **Easier Maintenance**: Independent updates for each image
5. **Proven Pattern**: Same approach as working `rec-sys-workflow`
