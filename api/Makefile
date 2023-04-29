# Pin the dependencies
lock:
	poetry lock

# Run the docker
run:
	poetry export -f requirements.txt --without-hashes --output requirements.txt
	docker-compose up -d --build

# Run the docker
stop:
	docker-compose down

# Run tests for the library
test:
	poetry export -f requirements.txt --without-hashes --output requirements.txt
	docker-compose up -d --build
	poetry export -f requirements.txt --without-hashes --dev --output requirements-dev.txt
	docker cp requirements-dev.txt api_backend_1:/app/requirements-dev.txt
	docker-compose exec -T backend pip install -r requirements-dev.txt
	docker cp tests api_backend_1:/app/tests
	docker-compose exec -T backend pytest tests/
	docker-compose down
