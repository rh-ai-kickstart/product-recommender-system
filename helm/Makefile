# Makefile for RAG Deployment
# Replaces the original deploy.sh script with additional uninstall functionality
ifeq ($(NAMESPACE),)
ifeq (,$(filter depend list-models% help,$(MAKECMDGOALS)))
$(error NAMESPACE is not set)
endif
endif

MAKEFLAGS += --no-print-directory

# Default values
POSTGRES_USER ?= postgres
POSTGRES_PASSWORD ?= recsys_password
POSTGRES_DBNAME ?= recsys
HF_TOKEN ?= $(shell bash -c 'read -r -p "Enter Hugging Face Token: " HF_TOKEN; echo $$HF_TOKEN')
HELM_CHART := product-recommender-system
TOLERATIONS_TEMPLATE=[{"key":"$(1)","effect":"NoSchedule","operator":"Exists"}]

helm_pgvector_args = \
    --set pgvector.secret.user=$(POSTGRES_USER) \
    --set pgvector.secret.password=$(POSTGRES_PASSWORD) \
    --set pgvector.secret.dbname=$(POSTGRES_DBNAME)

helm_llm_service_args = \
    --set llm-service.secret.hf_token=$(HF_TOKEN) \
    $(if $(DEVICE),--set llm-service.device='$(DEVICE)',) \
    $(if $(LLM),--set global.models.$(LLM).enabled=true,) \
    $(if $(SAFETY),--set global.models.$(SAFETY).enabled=true,) \
    $(if $(LLM_TOLERATION),--set-json global.models.$(LLM).tolerations='$(call TOLERATIONS_TEMPLATE,$(LLM_TOLERATION))',) \
    $(if $(SAFETY_TOLERATION),--set-json global.models.$(SAFETY).tolerations='$(call TOLERATIONS_TEMPLATE,$(SAFETY_TOLERATION))',) \
    $(if $(RAW_DEPLOYMENT),--set llm-service.rawDeploymentMode=$(RAW_DEPLOYMENT),)

helm_llama_stack_args = \
    $(if $(LLM),--set global.models.$(LLM).enabled=true,) \
    $(if $(SAFETY),--set global.models.$(SAFETY).enabled=true,) \
    $(if $(LLM_URL),--set global.models.$(LLM).url='$(LLM_URL)',) \
    $(if $(SAFETY_URL),--set global.models.$(SAFETY).url='$(SAFETY_URL)',) \
    $(if $(LLM_API_TOKEN),--set global.models.$(LLM).apiToken='$(LLM_API_TOKEN)',) \
    $(if $(SAFETY_API_TOKEN),--set global.models.$(SAFETY).apiToken='$(SAFETY_API_TOKEN)',) \
    $(if $(LLAMA_STACK_ENV),--set-json llama-stack.secrets='$(LLAMA_STACK_ENV)',) \
    $(if $(RAW_DEPLOYMENT),--set llama-stack.rawDeploymentMode=$(RAW_DEPLOYMENT),)

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  list-models   - List available models"
	@echo "  install       - Install the recommender system app (creates namespace, secrets, and deploys Helm chart)"
	@echo "  uninstall     - Uninstall the recommender system app and clean up resources"
	@echo "  status        - Check status of the deployment"
	@echo ""
	@echo "Configuration options (set via environment variables or make arguments):"
	@echo "  NAMESPACE                - Target namespace (default: llama-stack-rag)"
	@echo "  HF_TOKEN                 - Hugging Face Token (will prompt if not provided)"
	@echo "  DEVICE                	  - Deploy models on cpu or gpu (default)"
	@echo "  {SAFETY,LLM}             - Model id as defined in values (eg. llama-3-2-1b-instruct)"
	@echo "  {SAFETY,LLM}_URL         - Model URL"
	@echo "  {SAFETY,LLM}_API_TOKEN   - Model API token for remote models"
	@echo "  {SAFETY,LLM}_TOLERATION  - Model pod toleration"

# Create namespace and deploy
namespace:
	@oc create namespace $(NAMESPACE) &> /dev/null && oc label namespace $(NAMESPACE) modelmesh-enabled=false ||:
	@oc project $(NAMESPACE) &> /dev/null ||:

.PHONY: depend
depend:
	@echo "Updating Helm dependencies"
	@helm dependency update $(HELM_CHART) &> /dev/null

.PHONY: list-models
list-models: depend
	@helm template dummy-release $(HELM_CHART) --set llm-service._debugListModels=true | grep ^model:

.PHONY: install
install: namespace depend
	@$(eval PGVECTOR_ARGS := $(call helm_pgvector_args))
	@$(eval LLM_SERVICE_ARGS := $(call helm_llm_service_args))
	@$(eval LLAMA_STACK_ARGS := $(call helm_llama_stack_args))
	@echo "Installing $(HELM_CHART) helm chart"
	@helm -n $(NAMESPACE) upgrade --install $(HELM_CHART) $(HELM_CHART) -n $(NAMESPACE) \
		$(PGVECTOR_ARGS) \
		$(LLM_SERVICE_ARGS) \
		$(LLAMA_STACK_ARGS) \
		$(EXTRA_HELM_ARGS) --set strimzi-kafka-operator.createGlobalResources=false
	@echo "Waiting for model services and llamastack to deploy. It may take around 10-15 minutes depending on the size of the model..."
	@oc rollout status deploy
	@echo "$(HELM_CHART) installed successfully"

# Uninstall the deployment and clean up
.PHONY: uninstall
uninstall:
	@echo "Uninstalling $(HELM_CHART) helm chart"
	@helm -n $(NAMESPACE) uninstall $(HELM_CHART)
	@echo "Removing pgvector and minio PVCs from $(NAMESPACE)"
	@oc get pvc -n $(NAMESPACE) -o custom-columns=NAME:.metadata.name | grep -E '^(pg|minio)-data' | xargs -I {} oc delete pvc -n $(NAMESPACE) {} ||:
	@echo "Deleting remaining pods in namespace $(NAMESPACE)"
	@oc delete pods -n $(NAMESPACE) --all
	@echo "Checking for any remaining resources in namespace $(NAMESPACE)..."
	@echo "If you want to completely remove the namespace, run: oc delete project $(NAMESPACE)"
	@echo "Remaining resources in namespace $(NAMESPACE):"
	@$(MAKE) status

# Check deployment status
.PHONY: status
status:
	@echo "Listing pods..."
	oc get pods -n $(NAMESPACE) || true

	@echo "Listing services..."
	oc get svc -n $(NAMESPACE) || true

	@echo "Listing routes..."
	oc get routes -n $(NAMESPACE) || true

	@echo "Listing secrets..."
	oc get secrets -n $(NAMESPACE) | grep huggingface-secret || true

	@echo "Listing pvcs..."
	oc get pvc -n $(NAMESPACE) || true
