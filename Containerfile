# ---------- Frontend Build ----------
FROM registry.access.redhat.com/ubi9/nodejs-22 AS frontend-builder

USER root

WORKDIR /app/frontend

# Copy only package files first to leverage Docker cache
COPY frontend/package*.json ./

RUN npm install --debug

# Now copy the rest of the frontend code
COPY frontend/ ./

# Set node memory limit if needed
ENV NODE_OPTIONS=--max-old-space-size=2048

RUN npm run build

# ---------- Backend Build ----------
FROM quay.io/rh-ai-kickstart/recommendation-core:latest


USER root
WORKDIR /app/backend

COPY backend/pyproject.toml pyproject.toml
COPY recommendation-core/ /app/recommendation-core/

COPY backend/ ./

# Copy the frontend build output to backend/public
COPY --from=frontend-builder /app/frontend/dist ./public

# Set Hugging Face cache directory
ENV HF_HOME=/hf_cache

# Pre-download the model and fix permissions again (?) after download
RUN dnf update -y && pip3 install uv && uv pip install -r pyproject.toml && \
    mkdir -p /hf_cache && \
    chmod -R 777 /hf_cache && \
    python3 -c "from transformers import CLIPProcessor, CLIPModel; \
                CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
                CLIPModel.from_pretrained('openai/clip-vit-base-patch32')" && \
    chmod -R 777 /hf_cache && \
    chmod -R +r . 

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

WORKDIR /app/backend/src
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
