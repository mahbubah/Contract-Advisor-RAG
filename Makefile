# Define the default target.
all: build

# Build all services.
build:
	docker-compose build

# Start the services.
up:
	docker-compose up

# Stop the services.
down:
	docker-compose down

# Run the services in detached mode.
up-detached:
	docker-compose up -d

# Clean up Docker images and containers.
clean:
	docker-compose down --rmi all --volumes --remove-orphans

# Run tests (add this if you have tests defined)
test:
	docker-compose run backend pytest
