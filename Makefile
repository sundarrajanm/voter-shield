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

clean:
	docker rmi $(IMAGE) || true

### AWS ECR Related Targets ###

# can be comma separated list of S3 paths for multiple booths!
DRY_RUN_S3_INPUT=s3://264676382451-eci-download/dry-run/2025-EROLLGEN-S22-116-FinalRoll-Revision1-ENG-244-WI.pdf
DRY_RUN_S3_OUTPUT=s3://264676382451-eci-download/dry-run

ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | \
	docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

tag:
	docker tag $(IMAGE) $(ECR_REPO):$(IMAGE_TAG)

push: ecr-login tag
	docker push $(ECR_REPO):$(IMAGE_TAG)

run-fargate:
	./scripts/run-fargate.sh

run-fargate-dry-run:
	./scripts/run-fargate.sh \
	  --s3-input "$(DRY_RUN_S3_INPUT)" \
	  # --s3-output "$(DRY_RUN_S3_OUTPUT)" # Enable this after write permissions are granted

dry-run:
	python main.py --s3-input "$(DRY_RUN_S3_INPUT)" --s3-output "$(DRY_RUN_S3_OUTPUT)"

