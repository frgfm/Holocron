DOCKERFILE_PATH = ./api/Dockerfile
API_DIR = ./api
PKG_DIR = .
DEMO_DIR = ./demo
DOCS_DIR = ./docs
PKG_CONFIG_FILE = ${PKG_DIR}/pyproject.toml
PKG_TEST_DIR = ${PKG_DIR}/tests
API_CONFIG_FILE = ${API_DIR}/pyproject.toml
API_LOCK_FILE = ${API_DIR}/uv.lock
API_REQ_FILE = ${API_DIR}/requirements.txt
DEMO_REQ_FILE = ${DEMO_DIR}/requirements.txt
DEMO_SCRIPT = ${DEMO_DIR}/app.py
DOCKER_NAMESPACE ?= holocron
DOCKER_REPO ?= backend
DOCKER_TAG ?= latest

########################################################
# Code checks
########################################################


install-quality: ${PKG_CONFIG_FILE}
	uv pip install --system -e ".[quality]"
	pre-commit install

lint-check: ${PKG_CONFIG_FILE}
	ruff format --check . --config ${PKG_CONFIG_FILE}
	ruff check . --config ${PKG_CONFIG_FILE}

lint-format: ${PKG_CONFIG_FILE}
	ruff format . --config ${PKG_CONFIG_FILE}
	ruff check --fix . --config ${PKG_CONFIG_FILE}

precommit: ${PKG_CONFIG_FILE} .pre-commit-config.yaml
	pre-commit run --all-files

typing-check: ${PKG_CONFIG_FILE}
	mypy --config-file ${PKG_CONFIG_FILE}

deps-check: .github/verify_deps_sync.py
	python .github/verify_deps_sync.py

# this target runs checks on all files
quality: lint-check typing-check deps-check

style: lint-format precommit

########################################################
# Build
########################################################

# PACKAGE
install: ${PKG_CONFIG_FILE}
	uv pip install --system -e .

# TESTS
install-test: ${PKG_CONFIG_FILE}
	uv pip install --system -e ".[test]"

test: install-test ${PKG_TEST_DIR}
	pytest --cov=holocron tests/

# DEMO
install-demo: ${DEMO_REQ_FILE}
	uv pip install --system -r ${DEMO_REQ_FILE}

run-demo: install-demo ${DEMO_SCRIPT}
	python ${DEMO_SCRIPT} --port 8080

# DOCS
install-docs: ${PKG_CONFIG_FILE}
	uv pip install --system -e ".[docs]"

docs-latest: install-docs ${DOCS_DIR}
	sphinx-build ${DOCS_DIR}/source ${DOCS_DIR}/_build -a

docs-full: install-docs ${DOCS_DIR}
	cd ${DOCS_DIR} && bash build.sh

# API
lock: ${API_CONFIG_FILE}
	uv lock --project ${API_DIR}

req: ${API_CONFIG_FILE} ${PYTHON_LOCK_FILE}
	uv export --no-hashes --locked --no-dev -q -o ${API_REQ_FILE} --project ${API_DIR}

build-api: req ${DOCKERFILE_PATH}
	docker build --platform linux/amd64 ${API_DIR} -t ${DOCKER_NAMESPACE}/${DOCKER_REPO}:${DOCKER_TAG}

push-api: build-api
	docker push ${DOCKER_NAMESPACE}/${DOCKER_REPO}:${DOCKER_TAG}

# Run the docker
start-api: build-api ${API_DIR}/docker-compose.yml
	docker compose -f ${API_DIR}/docker-compose.yml up -d --wait

# Run the docker
stop-api: ${API_DIR}/docker-compose.yml
	docker compose -f ${API_DIR}/docker-compose.yml down
