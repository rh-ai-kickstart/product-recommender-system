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
# HF_TOKEN ?= $(shell bash -c 'read -r -p "Enter Hugging Face Token: " HF_TOKEN; echo $$HF_TOKEN')
HELM_CHART := product-recommender-system
TOLERATIONS_TEMPLATE=[{"key":"$(1)","effect":"NoSchedule","operator":"Exists"}]

helm_pgvector_args = \
    --set pgvector.secret.user=$(POSTGRES_USER) \
    --set pgvector.secret.password=$(POSTGRES_PASSWORD) \
    --set pgvector.secret.dbname=$(POSTGRES_DBNAME)

dataset_args = \
	--set datasetUrl=$(DATASET_URL)

# Shared function for Kafka topic cleanup
define cleanup_kafka_topics
	@echo "Cleaning up Kafka topics with graceful deletion..."
	@echo "Attempting graceful deletion of Kafka topics (60s grace period)..."
	@oc delete kafkatopic interactions -n $(NAMESPACE) --grace-period=60 ||:
	@oc delete kafkatopic new-users -n $(NAMESPACE) --grace-period=60 ||:
	@echo "Waiting for Kafka topics to be deleted..."
	@oc wait --for=delete kafkatopic/interactions -n $(NAMESPACE) --timeout=30s ||:
	@oc wait --for=delete kafkatopic/new-users -n $(NAMESPACE) --timeout=30s ||:
	@echo "Checking if Kafka topics still exist and removing finalizers if needed..."
	@if oc get kafkatopic interactions -n $(NAMESPACE) &>/dev/null; then \
		echo "Removing finalizers from interactions topic..."; \
		oc patch kafkatopic interactions -n $(NAMESPACE) -p '{"metadata":{"finalizers":[]}}' --type=merge ||:; \
	fi
	@if oc get kafkatopic new-users -n $(NAMESPACE) &>/dev/null; then \
		echo "Removing finalizers from new-users topic..."; \
		oc patch kafkatopic new-users -n $(NAMESPACE) -p '{"metadata":{"finalizers":[]}}' --type=merge ||:; \
	fi
endef

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  list-models   - List available models"
	@echo "  install       - Install the recommender system app (creates namespace, secrets, and deploys Helm chart)"
	@echo "  uninstall     - Uninstall the recommender system app and clean up resources"
	@echo "  delete-project - Delete the entire project with automatic Kafka topic cleanup"
	@echo "  status        - Check status of the deployment"
	@echo ""
	@echo "Configuration options (set via environment variables or make arguments):"
	@echo "  NAMESPACE                - Target namespace (default: llama-stack-rag)"

# Create namespace and deploy
namespace:
	@oc create namespace $(NAMESPACE) &> /dev/null && oc label namespace $(NAMESPACE) modelmesh-enabled=false ||:
	@oc project $(NAMESPACE) &> /dev/null ||:

.PHONY: depend
depend:
	@echo "Updating Helm dependencies"
	@helm dependency update $(HELM_CHART) &> /dev/null


.PHONY: install
install: namespace depend
	@$(eval PGVECTOR_ARGS := $(call helm_pgvector_args))
	@$(eval DATASET_ARGS := $(call dataset_args))
	@echo "Installing $(HELM_CHART) helm chart"
	@echo "Waiting for model services to deploy. It may take around 10-15 minutes depending on the size of the database..."
	@helm -n $(NAMESPACE) upgrade --install $(HELM_CHART) $(HELM_CHART) -n $(NAMESPACE) \
		$(PGVECTOR_ARGS) \
		$(DATASET_ARGS) \
		$(EXTRA_HELM_ARGS) --set strimzi-kafka-operator.createGlobalResources=false --timeout 300m
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
	@oc delete jobs -n $(NAMESPACE) --all
	@oc delete pods -n $(NAMESPACE) --all
	$(cleanup_kafka_topics)
	@echo "Checking for any remaining resources in namespace $(NAMESPACE)..."
	@echo "If you want to completely remove the namespace, run: make delete-project NAMESPACE=$(NAMESPACE)"
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

# Delete project with automatic Kafka topic cleanup
.PHONY: delete-project
delete-project:
	@echo "Deleting project $(NAMESPACE)..."
	@oc delete project $(NAMESPACE)
	@echo "Waiting 90 seconds to see if project deletion completes..."
	@sleep 90
	@if oc get project $(NAMESPACE) &>/dev/null; then \
		echo "Project deletion is taking longer than expected. Checking for stuck Kafka topics..."; \
		$(cleanup_kafka_topics); \
		echo "Project $(NAMESPACE) should now be deleted. If it's still stuck, check manually."; \
	else \
		echo "Project $(NAMESPACE) deleted successfully!"; \
	fi
