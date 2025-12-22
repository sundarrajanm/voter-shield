# =========================
# Project config
# =========================
APP_NAME := voter-shield
IMAGE_TAG := latest
IMAGE := $(APP_NAME):$(IMAGE_TAG)

AWS_REGION := ap-south-1
AWS_ACCOUNT_ID := 123456789012
ECR_REPO := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP_NAME)

setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest -m regression -v -s

run:
	python main.py

build:
	docker build -t $(IMAGE) .

samplerun:
	docker run --rm \
		$(IMAGE) \
		python3 main.py \
		--constituency-number 999 \
		--batch-no B01 \
		--pdf-files s3://264676382451-eci-download/sample/sample_tamil.pdf

shell:
	docker run --rm -it $(IMAGE) bash

ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | \
	docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

tag:
	docker tag $(IMAGE) $(ECR_REPO):$(IMAGE_TAG)

push: ecr-login tag
	docker push $(ECR_REPO):$(IMAGE_TAG)

clean:
	docker rmi $(IMAGE) || true