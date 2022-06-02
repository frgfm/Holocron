# this target runs checks on all files
quality:
	isort . -c -v
	flake8 ./
	mypy holocron/
	pydocstyle holocron/
	black --check .

# this target runs checks on all files and potentially modifies some of them
style:
	isort .
	black .

# Run tests for the library
test:
	coverage run -m pytest tests/


# Run sanity checks for the library
sanity:
	python .github/validate_headers.py

# Build documentation for current version
docs-single-version:
	sphinx-build docs/source docs/_build -a

# Check that docs can build
docs:
	cd docs && bash build.sh

