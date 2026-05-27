.PHONY: help install run test lint format clean docker-build docker-run

help:
	@echo "Fatocheck - Fake News Detection API"
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make run           - Run API server"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format code"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make clean         - Clean up cache files"

install:
	pip install -r requirements.txt

run:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	flake8 . --max-line-length=120
	pylint api/ utils/

format:
	black . --line-length=120
	isort .

docker-build:
	docker build -t fatocheck:latest .

docker-run:
	docker run -p 8000:8000 fatocheck:latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
