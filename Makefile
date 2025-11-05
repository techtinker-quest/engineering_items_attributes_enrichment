.PHONY: install test clean lint format

install:
	pip install -e .
	pip install -r deployment/requirements-dev.txt

test:
	bash scripts/run_tests.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf dist/ build/ *.egg-info htmlcov/

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/