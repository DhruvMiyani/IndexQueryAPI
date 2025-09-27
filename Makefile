# Makefile for Vector Database API
# Run 'make help' for available commands

.PHONY: help install run test docker-build docker-run docker-up docker-down clean format lint

# Variables
PYTHON := python3
PIP := pip
DOCKER_IMAGE := vector-db-api
DOCKER_TAG := latest
PORT := 8000
COHERE_API_KEY := pa6sRhnVAedMVClPAwoCvC1MjHKEwjtcGSTjWRMd

# Default target
help: ## Show this help message
	@echo "Vector Database API - Available Commands:"
	@echo "========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development Setup
install: ## Install dependencies
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && $(PIP) install --upgrade pip
	. .venv/bin/activate && $(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed. Activate venv with: source .venv/bin/activate"

dev-install: install ## Install dev dependencies
	. .venv/bin/activate && $(PIP) install pytest pytest-cov pytest-asyncio black ruff mypy
	@echo "✅ Development dependencies installed"

# Running the Application
run: ## Run the application locally
	@export COHERE_API_KEY=$(COHERE_API_KEY) && \
	cd src && $(PYTHON) main.py

run-dev: ## Run with auto-reload for development
	@export COHERE_API_KEY=$(COHERE_API_KEY) && \
	cd src && uvicorn main:app --reload --host 0.0.0.0 --port $(PORT)

# Testing
test: ## Run all tests
	. .venv/bin/activate && pytest tests/ -v

test-coverage: ## Run tests with coverage report
	. .venv/bin/activate && pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "📊 Coverage report generated in htmlcov/index.html"

test-integration: ## Run integration tests only
	. .venv/bin/activate && pytest tests/test_api_integration.py -v

test-unit: ## Run unit tests only
	. .venv/bin/activate && pytest tests/ -v -k "not integration"

# Code Quality
format: ## Format code with black
	. .venv/bin/activate && black src/ tests/
	@echo "✨ Code formatted"

lint: ## Lint code with ruff
	. .venv/bin/activate && ruff check src/ tests/
	@echo "🔍 Linting complete"

typecheck: ## Type check with mypy
	. .venv/bin/activate && mypy src/
	@echo "✅ Type checking complete"

quality: format lint ## Run all code quality checks
	@echo "✅ All code quality checks passed"

# Docker Commands
docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "🐳 Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)"

docker-run: ## Run Docker container
	docker run -d \
		--name $(DOCKER_IMAGE) \
		-p $(PORT):8000 \
		-e COHERE_API_KEY=$(COHERE_API_KEY) \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "🚀 Container running at http://localhost:$(PORT)"

docker-stop: ## Stop Docker container
	docker stop $(DOCKER_IMAGE) || true
	docker rm $(DOCKER_IMAGE) || true
	@echo "🛑 Container stopped and removed"

docker-logs: ## Show Docker container logs
	docker logs -f $(DOCKER_IMAGE)

docker-shell: ## Open shell in Docker container
	docker exec -it $(DOCKER_IMAGE) bash

# Docker Compose Commands
docker-up: ## Start services with docker-compose
	docker-compose up -d
	@echo "🚀 Services started at http://localhost:$(PORT)"

docker-down: ## Stop services with docker-compose
	docker-compose down
	@echo "🛑 Services stopped"

docker-restart: docker-down docker-up ## Restart all services
	@echo "🔄 Services restarted"

docker-dev: ## Start development environment with docker-compose
	docker-compose --profile dev up -d vector-db-api-dev
	@echo "🚀 Development server at http://localhost:8001"

# API Testing Commands
api-health: ## Check API health
	@curl -s http://localhost:$(PORT)/health | python -m json.tool

api-create-library: ## Create a test library
	@curl -X POST http://localhost:$(PORT)/libraries \
		-H "Content-Type: application/json" \
		-d '{"name":"Test Library","index_type":"linear"}' | python -m json.tool

api-test-flow: ## Run complete test flow
	@echo "Testing API flow..."
	@$(MAKE) api-health
	@$(MAKE) api-create-library
	@echo "✅ API test flow complete"

# Cleanup
clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ htmlcov/ .coverage
	rm -rf dist/ build/ *.egg-info/
	@echo "🧹 Cleanup complete"

clean-docker: ## Clean Docker resources
	docker stop $(DOCKER_IMAGE) 2>/dev/null || true
	docker rm $(DOCKER_IMAGE) 2>/dev/null || true
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true
	docker system prune -f
	@echo "🧹 Docker cleanup complete"

clean-all: clean clean-docker ## Clean everything
	rm -rf .venv/
	@echo "🧹 Full cleanup complete"

# Documentation
docs: ## Open API documentation in browser
	@echo "Opening API docs at http://localhost:$(PORT)/docs"
	@open http://localhost:$(PORT)/docs || xdg-open http://localhost:$(PORT)/docs

# Utility Commands
env-setup: ## Create .env file from example
	cp .env.example .env
	@echo "📝 .env file created. Edit it to set your configuration."

check-deps: ## Check if all dependencies are installed
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "❌ Python not installed"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker not installed"; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "❌ docker-compose not installed"; exit 1; }
	@echo "✅ All dependencies installed"

# Quick Start Commands
quickstart: install docker-build docker-up ## Quick start for development
	@echo "🎉 Vector Database API is ready!"
	@echo "Local: http://localhost:$(PORT)/docs"
	@echo "Run 'make help' to see all available commands"

demo: docker-up api-test-flow ## Run a quick demo
	@echo "🎉 Demo complete! API is running at http://localhost:$(PORT)/docs"