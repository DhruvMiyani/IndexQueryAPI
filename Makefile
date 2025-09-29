# Vector Database API - Docker Makefile

# Variables
IMAGE_NAME = vectordb-api
IMAGE_TAG = latest
CONTAINER_NAME = vectordb
PORT = 8000
VOLUME_NAME = vectordb_data

# Help
.PHONY: help
help: ## Show this help message
	@echo "Vector Database API - Docker Commands"
	@echo "====================================="
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# Build Commands
.PHONY: build
build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

.PHONY: build-no-cache
build-no-cache: ## Build Docker image without cache
	docker build --no-cache -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Run Commands
.PHONY: run
run: ## Run container
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		-v $(VOLUME_NAME):/app/data \
		-e COHERE_API_KEY=${COHERE_API_KEY} \
		$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: run-dev
run-dev: ## Run container in development mode
	docker run -it --rm \
		--name $(CONTAINER_NAME)-dev \
		-p $(PORT):8000 \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/sdk:/app/sdk \
		-e COHERE_API_KEY=${COHERE_API_KEY} \
		$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: run-interactive
run-interactive: ## Run container interactively
	docker run -it --rm \
		--name $(CONTAINER_NAME)-interactive \
		-p $(PORT):8000 \
		--entrypoint /bin/bash \
		$(IMAGE_NAME):$(IMAGE_TAG)

# Docker Compose Commands
.PHONY: up
up: ## Start services with docker-compose
	docker-compose up -d

.PHONY: up-prod
up-prod: ## Start services with production profile
	docker-compose --profile production up -d

.PHONY: down
down: ## Stop and remove containers
	docker-compose down

.PHONY: restart
restart: down up ## Restart services

# Management Commands
.PHONY: logs
logs: ## Show container logs
	docker logs -f $(CONTAINER_NAME)

.PHONY: logs-compose
logs-compose: ## Show docker-compose logs
	docker-compose logs -f

.PHONY: shell
shell: ## Open shell in running container
	docker exec -it $(CONTAINER_NAME) /bin/bash

.PHONY: stop
stop: ## Stop container
	docker stop $(CONTAINER_NAME)

.PHONY: start
start: ## Start stopped container
	docker start $(CONTAINER_NAME)

.PHONY: remove
remove: stop ## Remove container
	docker rm $(CONTAINER_NAME)

# Testing Commands
.PHONY: test
test: ## Test the API endpoints
	@echo "Testing health endpoint..."
	curl -f http://localhost:$(PORT)/health || echo "Health check failed"
	@echo "\nTesting library creation..."
	curl -X POST "http://localhost:$(PORT)/libraries" \
		-H "Content-Type: application/json" \
		-d '{"name": "Test Library", "metadata": {"test": true}}' || echo "Library creation failed"
	@echo "\nTesting library listing..."
	curl -f http://localhost:$(PORT)/libraries || echo "Library listing failed"

.PHONY: test-sdk
test-sdk: ## Test SDK functionality
	docker exec $(CONTAINER_NAME) python3 /app/sdk/examples.py

# Cleanup Commands
.PHONY: clean
clean: ## Remove container and image
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	-docker rmi $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: clean-all
clean-all: clean ## Remove everything including volumes
	-docker volume rm $(VOLUME_NAME)
	-docker system prune -f

# Volume Management
.PHONY: volume-create
volume-create: ## Create data volume
	docker volume create $(VOLUME_NAME)

.PHONY: volume-backup
volume-backup: ## Backup data volume
	docker run --rm \
		-v $(VOLUME_NAME):/data \
		-v $(PWD):/backup \
		alpine tar czf /backup/vectordb-backup-$(shell date +%Y%m%d_%H%M%S).tar.gz -C /data .

.PHONY: volume-restore
volume-restore: ## Restore data volume (requires BACKUP_FILE)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Please specify BACKUP_FILE=<file>"; exit 1; fi
	docker run --rm \
		-v $(VOLUME_NAME):/data \
		-v $(PWD):/backup \
		alpine tar xzf /backup/$(BACKUP_FILE) -C /data

# Monitoring Commands
.PHONY: stats
stats: ## Show container resource usage
	docker stats $(CONTAINER_NAME)

.PHONY: inspect
inspect: ## Inspect container configuration
	docker inspect $(CONTAINER_NAME)

.PHONY: health
health: ## Check container health status
	docker inspect $(CONTAINER_NAME) --format='{{.State.Health.Status}}'

# Development Commands
.PHONY: dev-setup
dev-setup: volume-create build run ## Complete development setup

.PHONY: dev-reset
dev-reset: clean-all dev-setup ## Reset development environment

.PHONY: quick-test
quick-test: build run ## Quick build and test
	sleep 5
	$(MAKE) test
	$(MAKE) remove

# Production Commands
.PHONY: prod-deploy
prod-deploy: ## Deploy to production
	$(MAKE) build
	$(MAKE) up-prod

.PHONY: prod-update
prod-update: ## Update production deployment
	docker-compose pull
	docker-compose up -d --force-recreate

# Debug Commands
.PHONY: debug
debug: ## Debug container issues
	@echo "=== Container Status ==="
	docker ps -a | grep $(CONTAINER_NAME) || echo "Container not found"
	@echo "\n=== Resource Usage ==="
	docker stats --no-stream $(CONTAINER_NAME) 2>/dev/null || echo "Container not running"
	@echo "\n=== Recent Logs ==="
	docker logs --tail 20 $(CONTAINER_NAME) 2>/dev/null || echo "No logs available"
	@echo "\n=== Health Check ==="
	curl -f http://localhost:$(PORT)/health 2>/dev/null || echo "Health check failed"

# Multi-platform build (for deployment)
.PHONY: build-multiplatform
build-multiplatform: ## Build for multiple platforms
	docker buildx build --platform linux/amd64,linux/arm64 -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Default target
.DEFAULT_GOAL := help