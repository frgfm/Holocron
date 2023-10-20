# Pin the dependencies
lock:
	poetry lock

build:
	poetry export -f requirements.txt --without-hashes --output requirements.txt
	docker build . -t frgfm/holocron:python3.9-slim

# Run the docker
run:
	poetry export -f requirements.txt --without-hashes --output requirements.txt
	docker compose up -d --build

# Run the docker
stop:
	docker compose down

# Run tests for the library
test:
	poetry export -f requirements.txt --without-hashes --with dev --output requirements.txt
	docker compose up -d --build
	docker compose exec -T backend pytest tests/ --cov=app
	docker compose down
