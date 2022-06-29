# Pin the dependencies
lock:
	poetry lock
	poetry export -f requirements.txt --without-hashes --output requirements.txt
	poetry export -f requirements.txt --without-hashes --dev --output requirements-dev.txt

# Run the docker
run:
	docker-compose up -d --build

# Run the docker
stop:
	docker-compose down

# Run tests for the library
test:
	docker-compose up -d --build
	docker cp requirements-dev.txt api_holocron_1:/app/requirements-dev.txt
	docker-compose exec -T holocron pip install -r requirements-dev.txt
	docker cp tests api_holocron_1:/app/tests
	docker-compose exec -T holocron pytest tests/
	docker-compose down
