.PHONY: lint test format check help

# Default target
help:
	@echo "Available targets:"
	@echo "  make lint      - Run ruff and mypy checks"
	@echo "  make format    - Format code with ruff"
	@echo "  make test      - Run tests (renders a sample scene)"
	@echo "  make check     - Run format, lint, and test"
	@echo "  make help      - Show this help message"

# Lint with ruff and mypy
lint:
	@echo "Running ruff linter..."
	uv run ruff check --fix understanding_muon/
	@echo "Running mypy type checker..."
	uv run mypy understanding_muon/

# Format code with ruff
format:
	@echo "Formatting code with ruff..."
	uv run ruff format understanding_muon/
	uv run ruff check --fix understanding_muon/

# Test by rendering a sample scene
test:
	@echo "Testing by rendering a sample scene..."
	uv run manim -pql understanding_muon/chapter_01_intro.py MLTrainingProcess

# Run all checks
check: format lint test
	@echo "All checks passed!"
