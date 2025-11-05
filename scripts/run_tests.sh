#!/bin/bash
# Run test suite with coverage

echo "Running Drawing Intelligence System Tests"
echo "=========================================="

# Run unit tests
echo -e "\n📦 Running unit tests..."
pytest tests/unit/ -v --cov=src/drawing_intelligence --cov-report=term-missing

# Run integration tests
echo -e "\n🔗 Running integration tests..."
pytest tests/integration/ -v -m integration

# Run performance tests
echo -e "\n⚡ Running performance tests..."
pytest tests/performance/ -v -m "not slow"

# Generate HTML coverage report
echo -e "\n📊 Generating coverage report..."
pytest tests/ --cov=src/drawing_intelligence --cov-report=html

echo -e "\n✅ Tests complete! Coverage report: htmlcov/index.html"