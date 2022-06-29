# this target runs checks on all files
quality:
	isort . -c
	flake8
	mypy
	pydocstyle
	black --check .

# this target runs checks on all files and potentially modifies some of them
style:
	isort .
	black .

# Run tests for the library
test:
	coverage run -m pytest tests/

# Build documentation for current version
docs:
	sphinx-build docs/source docs/_build -a

# Check that docs can build
full-docs:
	cd docs && bash build.sh

# Run the Gradio demo
run-demo:
	python demo/app.py --port 8080

# Build the docker
docker:
	docker build . -t holocron:python3.8.1-slim

# Pin the dependencies
lock-api:
	cd api && poetry lock
	cd api && poetry export -f requirements.txt --without-hashes --output requirements.txt
	cd api && poetry export -f requirements.txt --without-hashes --dev --output requirements-dev.txt

# Run the docker
run-api:
	docker-compose up -d --build

# Run the docker
stop-api:
	docker-compose down

# Run tests for the library
test-api:
	docker-compose up -d --build
	docker cp api/requirements-dev.txt holocron_api_1:/app/requirements-dev.txt
	docker-compose exec -T api pip install -r requirements-dev.txt
	docker cp api/tests holocron_api_1:/app/tests
	docker-compose exec -T api pytest tests/
	docker-compose down
