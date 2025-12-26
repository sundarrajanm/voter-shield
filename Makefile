include config.mk

.PHONY: setup check-quality run build build-on-mac run-dev-docker ecr-login tag push run-fargate clean

setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

check-quality:
	./scripts/quality.sh

run:
	python main.py --delete-old

build:
	docker build -t $(IMAGE) .

build-on-mac:
	DOCKER_BUILDKIT=0 docker build \
		--platform linux/amd64 \
		-t $(IMAGE) \
		.

run-dev-docker:
	time docker run --rm \
		--cpus=1 \
		--memory=4g \
		-v .:/app \
		$(IMAGE) \
		--delete-old

ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | \
	docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

tag:
	docker tag $(IMAGE) $(ECR_REPO):$(IMAGE_TAG)

push: ecr-login tag
	docker push $(ECR_REPO):$(IMAGE_TAG)

run-fargate:
	./scripts/run-fargate.sh

clean:
	docker rmi $(IMAGE) || true
