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
FROM registry.access.redhat.com/ubi9/python-312

USER root
WORKDIR /app/backend

RUN dnf update -y

# Install uv and install dependencies
RUN pip3 install uv
COPY backend/pyproject.toml pyproject.toml
RUN uv pip install -r pyproject.toml

# Copy the backend code
COPY backend/ ./

# Copy the frontend build output to backend/public
COPY --from=frontend-builder /app/frontend/dist ./public

# Set Hugging Face cache directory
ENV HF_HOME=/hf_cache
RUN mkdir -p /hf_cache && \
    chmod -R 777 /hf_cache

# Pre-download the model
RUN python3 -c "from transformers import CLIPProcessor, CLIPModel; \
                CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
                CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"

# Fix permissions again after download
RUN chmod -R 777 /hf_cache
RUN chmod -R +r . && ls -la

EXPOSE 8000
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
