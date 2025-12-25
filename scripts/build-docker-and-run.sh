#!/usr/bin/env bash
set -e

IMAGE="votershield-calib"

echo "🐳 Building image..."
docker build -t $IMAGE .

echo "🚀 Running container..."
docker run --rm \
  --cpus=1 \
  --memory=1g \
  $IMAGE "$@"
