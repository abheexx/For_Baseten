#!/bin/bash

# Test runner script for Whisper Inference Service
set -e

echo "🧪 Running Whisper Inference Service Tests"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider running: python -m venv venv && source venv/bin/activate"
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run linting
echo "🔍 Running linting checks..."
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Check code formatting
echo "🎨 Checking code formatting..."
black --check --diff .
isort --check-only --diff .

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v --cov=. --cov-report=xml --cov-report=html

# Run load tests (optional)
if command -v k6 &> /dev/null; then
    echo "⚡ Running load tests..."
    # Start service in background
    python main.py &
    SERVICE_PID=$!
    
    # Wait for service to start
    sleep 10
    
    # Run load test
    k6 run k6-load.js
    
    # Stop service
    kill $SERVICE_PID
else
    echo "⚠️  k6 not found, skipping load tests"
    echo "   Install k6 to run load tests: https://k6.io/docs/getting-started/installation/"
fi

echo "✅ All tests completed successfully!"
