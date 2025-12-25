#!/usr/bin/env bash
set -e

echo "🔍 Running Ruff (lint + autofix)..."
ruff check . --fix

echo "🎨 Running Black (format)..."
black .

echo "🧪 Running tests..."
pytest -q -v -s -ra --disable-warnings

echo "✅ Quality gate passed"
