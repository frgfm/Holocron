PY_DIR = .
DEMO_DIR = ./demo
TESTS_DIR = ./tests
DOCS_DIR = ./docs
BACKEND_DIR = ./api
SCRIPTS_DIR = ./scripts
PACKAGE_DIR = ${PY_DIR}/holocron
PYPROJECT_FILE = ${PY_DIR}/pyproject.toml
PYTHON_REQ_FILE = /tmp/requirements.txt

BACKEND_DOCKERFILE = ${BACKEND_DIR}/Dockerfile
PKG_TEST_DIR = ${PY_DIR}/tests
API_CONFIG_FILE = ${BACKEND_DIR}/pyproject.toml
API_LOCK_FILE = ${BACKEND_DIR}/uv.lock
API_REQ_FILE = ${BACKEND_DIR}/requirements.txt
DEMO_REQ_FILE = ${DEMO_DIR}/requirements.txt
DEMO_FILE = ${DEMO_DIR}/app.py
LATENCY_SCRIPT = ${SCRIPTS_DIR}/eval_latency.py
LATENCY_ARCH ?= rexnet1_0x
LATENCY_ARGS ?=
REPO_OWNER ?= frgfm
REPO_NAME ?= holocron
DOCKER_NAMESPACE ?= ghcr.io/${REPO_OWNER}
DOCKER_TAG ?= latest
DOCKER_PLATFORM ?= linux/amd64
PYTHON_REQ_FILE = /tmp/requirements.txt

.PHONY: help install install-quality lint-check lint-format headers-check headers-fix precommit typing-check deps-check quality style init-gh-labels init-gh-settings install-mintlify start-mintlify

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

########################################################
# Install & Setup
########################################################

venv:
	uv venv --python 3.11

install: ${PY_DIR} ${PYPROJECT_FILE} ## Install the core library
	uv sync --locked --no-dev

set-version: ${GLOBAL_PYPROJECT} ${BACKEND_PYPROJECT} ## Set the version in the pyproject.toml file
	uv version --frozen --no-build ${BUILD_VERSION}
	uv version --frozen --no-build --project ${BACKEND_DIR} ${BUILD_VERSION}

########################################################
# Code checks
########################################################


install-quality: ${PY_DIR} ${PYPROJECT_FILE} ## Install with quality dependencies
	uv sync --locked --extra quality

lint-check: ${PYPROJECT_FILE} ## Check code formatting and linting
	ruff check . --config ${PYPROJECT_FILE}
	ruff format --check . --config ${PYPROJECT_FILE}

lint-format: ${PYPROJECT_FILE} ## Format code and fix linting issues
	ruff check --fix . --config ${PYPROJECT_FILE}
	ruff format . --config ${PYPROJECT_FILE}

precommit: ${PYPROJECT_FILE} .pre-commit-config.yaml ## Run pre-commit hooks
	prek run --all-files

typing-check: ${PYPROJECT_FILE} ## Check type annotations
	uv run --no-sync ty check .

headers-check: ${PYPROJECT_FILE} ## Check Python copyright and license headers
	uv run --no-sync lmh check

headers-fix: ${PYPROJECT_FILE} ## Refresh recognized stale copyright years
	uv run --locked --extra quality lmh fix

deps-check: .github/verify_deps_sync.py ## Check dependency synchronization
	uv run --script .github/verify_deps_sync.py

# this target runs checks on all files
quality: lint-check typing-check headers-check deps-check ## Run all quality checks

style: precommit ## Format code and run pre-commit hooks

########################################################
# Builds
########################################################

build: ${PYPROJECT_FILE} ## Build the package
	uv build ${PY_DIR}

publish: ${PY_DIR} ## Publish the package to PyPI
	uv publish --trusted-publishing always

########################################################
# Tests
########################################################

install-test: ${PY_DIR} ${PYPROJECT_FILE} ## Install with test dependencies
	uv sync --locked --extra test

test: ${PYPROJECT_FILE} ## Run the tests
	uv run --no-sync pytest --cov-report xml

########################################################
# Scripts
########################################################

install-scripts: ${PY_DIR} ${PYPROJECT_FILE} ## Install with test dependencies
	uv sync --locked --extra scripts

bench-latency: ${PYPROJECT_FILE} ${LATENCY_SCRIPT} ## Measure model inference latency, throughput, and peak RSS
	uv run --no-sync python ${LATENCY_SCRIPT} ${LATENCY_ARCH} ${LATENCY_ARGS}


########################################################
# Docs
########################################################

install-docs: ${PYPROJECT_FILE}
	uv sync --locked --extra docs

# Build documentation for current version
serve-docs: ${DOCS_DIR}
	DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run mkdocs serve -f ${DOCS_DIR}/mkdocs.yml

# Check that docs can build
build-docs: ${DOCS_DIR}
	DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run --no-sync mkdocs build -f ${DOCS_DIR}/mkdocs.yml

push-docs: ${DOCS_DIR}
	DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run mkdocs gh-deploy -f ${DOCS_DIR}/mkdocs.yml --force

########################################################
# Demo
########################################################

install-demo: ${DEMO_FILE}
	uv sync --script ${DEMO_FILE}

# Run the Gradio demo
run-demo: ${DEMO_FILE}
	uv run --script ${DEMO_FILE} --port 3000

########################################################
# Backend
########################################################

lock-backend: ${BACKEND_DIR} ${BACKEND_PYPROJECT} ## Lock the backend dependencies
	uv lock --project ${BACKEND_DIR}

install-backend: ${BACKEND_DIR} ${BACKEND_PYPROJECT} ## Install the backend deps
	uv --project ${BACKEND_DIR} sync --locked --no-dev  --no-install-project

install-backend-test: ${BACKEND_DIR} ${BACKEND_PYPROJECT} ## Install the backend deps
	uv --project ${BACKEND_DIR} sync --locked --no-dev --extra test --no-install-project

import-backend: ${BACKEND_DIR} ${BACKEND_PYPROJECT} ## Install the backend deps
	uv export --project ${BACKEND_DIR} --no-hashes --locked --no-dev -o ${PYTHON_REQ_FILE}
	uv pip install -r ${PYTHON_REQ_FILE}

build-backend: ${BACKEND_DIR} ${BACKEND_DOCKERFILE} ## Build the backend container
	docker buildx build --platform ${DOCKER_PLATFORM} -f ${BACKEND_DOCKERFILE} -t ${DOCKER_NAMESPACE}/${REPO_NAME}-backend:${DOCKER_TAG} ${BACKEND_DIR}

push-backend: build-backend
	docker push ${DOCKER_NAMESPACE}/${REPO_NAME}-backend:${DOCKER_TAG}

########################################################
# Run backend
########################################################

uvicorn-backend: ${BACKEND_DIR}
	uv --project ${BACKEND_DIR} run uvicorn app.main:app --reload --reload-dir ${BACKEND_DIR} --host 0.0.0.0 --port 8080 --proxy-headers --use-colors --log-level info --app-dir ${BACKEND_DIR}

start-backend: build-backend ${BACKEND_DIR}
	docker run -p 8080:8080 ${DOCKER_NAMESPACE}/${REPO_NAME}-backend:${DOCKER_TAG}

stop-backend: ${BACKEND_DIR}
	docker stop ${DOCKER_NAMESPACE}/${REPO_NAME}-backend:${DOCKER_TAG}

test-backend:  ${BACKEND_DIR}/tests
	uv --directory ${BACKEND_DIR} run pytest
