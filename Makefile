# this target runs checks on all files
quality:
    isort **/*.py -c -v
    flake8 ./
    mypy holocron/

# this target runs checks on all files and potentially modifies some of them
style:
    isort **/*.py --skip "**/__init_.py" --filter-file

# Run tests for the library
test:
    coverage run -m pytest tests/

# Check that docs can build
docs:
    cd docs && bash build.sh