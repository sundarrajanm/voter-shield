# =========================
# Project config
# =========================
APP_NAME := voter-shield
IMAGE_TAG := latest
IMAGE := $(APP_NAME):$(IMAGE_TAG)

AWS_REGION := ap-south-1
AWS_ACCOUNT_ID := 264676382451
ECR_REPO := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP_NAME)

# ECS config
ECS_CLUSTER := default
TASK_FAMILY := voter-shield-task
CONTAINER_NAME := voter-shield

SUBNET_ID := subnet-004483a36813c238e
SECURITY_GROUP := sg-0a47febe680b97754

SAMPLE_PDF := s3://264676382451-eci-download/sample/sample_tamil.pdf

# =========================
# Local
# =========================

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
		--pdf-files $(SAMPLE_PDF)

shell:
	docker run --rm -it $(IMAGE) bash

# =========================
# AWS / ECR
# =========================

ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | \
	docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

create-ecr:
	aws ecr create-repository \
		--repository-name $(APP_NAME) \
		--region $(AWS_REGION)

tag:
	docker tag $(IMAGE) $(ECR_REPO):$(IMAGE_TAG)

push: ecr-login tag
	docker push $(ECR_REPO):$(IMAGE_TAG)

# =========================
# AWS / ECS
# =========================
ecs-create-cluster:
	aws ecs create-cluster \
		--cluster-name default \
		--region $(AWS_REGION)

ecs-register:
	aws ecs register-task-definition \
		--cli-input-json file://infra/voter-shield-task.json \
		--region $(AWS_REGION)

ecs-testrun:
	aws ecs run-task \
	  --cluster $(ECS_CLUSTER) \
	  --capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1 \
	  --task-definition $(TASK_FAMILY) \
	  --count 1 \
	  --network-configuration "awsvpcConfiguration={subnets=[$(SUBNET_ID)],securityGroups=[$(SECURITY_GROUP)],assignPublicIp=ENABLED}" \
	  --overrides '{"containerOverrides":[{"name":"$(CONTAINER_NAME)","command":["python","main.py","--constituency-number","999","--batch-no","B01","--pdf-files","$(SAMPLE_PDF)"]}]}' \
	  --region $(AWS_REGION)

# =========================
# Cleanup
# =========================

clean:
	docker rmi $(IMAGE) || true