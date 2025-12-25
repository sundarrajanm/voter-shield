#!/usr/bin/env bash
set -e

echo "🔍 Running Ruff (lint + autofix)..."
ruff check . --fix

echo "🎨 Running Black (format)..."
black .

echo "🧪 Running tests..."
pytest -q

echo "✅ Quality gate passed"
