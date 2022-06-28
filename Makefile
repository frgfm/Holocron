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
docs-single-version:
	sphinx-build docs/source docs/_build -a

# Check that docs can build
docs:
	cd docs && bash build.sh

# Run the Gradio demo
run-demo:
	python demo/app.py --port 8080

# Build the docker
docker-package:
	docker build . -t holocron:python3.8.1-slim

# Run the docker
run-api:
	docker-compose up -d --build

# Run the docker
stop-api:
	docker-compose down

# Run tests for the library
test-api:
	docker-compose up -d --build
	docker-compose exec -T holocron pip install -r requirements-dev.txt
	docker-compose exec -T holocron pytest tests/
	docker-compose down