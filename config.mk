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
