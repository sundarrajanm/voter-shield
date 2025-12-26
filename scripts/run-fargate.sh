#!/usr/bin/env bash
set -euo pipefail

echo "🛡️ VoterShield – ECS Fargate Runner"
echo "----------------------------------"

# ---------- Defaults ----------
DEFAULT_PROFILE="voter-shield"
LOG_GROUP="/ecs/voter-shield"

# ---------- Hardcoded infra ----------
export ECS_CLUSTER="default"
export TASK_FAMILY="voter-shield-task"
export SUBNET_ID="subnet-004483a36813c238e"
export SECURITY_GROUP="sg-0a47febe680b97754"
export AWS_REGION="ap-south-1"
export CONTAINER_NAME="voter-shield"
# ----------------------------------

# ---------- Validate ----------
: "${ECS_CLUSTER:?Missing ECS_CLUSTER}"
: "${TASK_FAMILY:?Missing TASK_FAMILY}"
: "${SUBNET_ID:?Missing SUBNET_ID}"
: "${SECURITY_GROUP:?Missing SECURITY_GROUP}"
: "${AWS_REGION:?Missing AWS_REGION}"

# ---------- AWS profile ----------
if [[ -t 0 ]]; then
  read -rp "AWS profile [$DEFAULT_PROFILE]: " AWS_PROFILE
  AWS_PROFILE=${AWS_PROFILE:-$DEFAULT_PROFILE}
else
  AWS_PROFILE=${AWS_PROFILE:-$DEFAULT_PROFILE}
fi

echo "🔑 Using AWS profile: $AWS_PROFILE"

# ---------- Container args ----------
CONTAINER_ARGS=("$@")

if [[ ${#CONTAINER_ARGS[@]} -gt 0 ]]; then
  echo "📦 Container arguments:"
  printf '  %q\n' "${CONTAINER_ARGS[@]}"
else
  echo "📦 No container arguments provided"
fi

# ---------- Build overrides JSON ----------
OVERRIDES_JSON=$(jq -n \
  --arg name "$CONTAINER_NAME" \
  --argjson args "$(printf '%s\n' "${CONTAINER_ARGS[@]}" | jq -R . | jq -s .)" \
  '{
    containerOverrides: [
      {
        name: $name,
        command: $args
      }
    ]
  }')

# ---------- Run ECS task ----------
echo "🚀 Launching ECS task..."

TASK_ARN=$(aws ecs run-task \
  --profile "$AWS_PROFILE" \
  --cluster "$ECS_CLUSTER" \
  --capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1 \
  --task-definition "$TASK_FAMILY" \
  --count 1 \
  --overrides "$OVERRIDES_JSON" \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$SECURITY_GROUP],assignPublicIp=ENABLED}" \
  --region "$AWS_REGION" \
  --query "tasks[0].taskArn" \
  --output text)

if [[ -z "$TASK_ARN" || "$TASK_ARN" == "None" ]]; then
  echo "❌ Failed to start ECS task"
  exit 1
fi

TASK_ID=$(basename "$TASK_ARN")
echo "🆔 Task ID: $TASK_ID"

# ---------- Wait for RUNNING ----------
echo "⏳ Waiting for task to start..."

while true; do
  STATUS=$(aws ecs describe-tasks \
    --profile "$AWS_PROFILE" \
    --cluster "$ECS_CLUSTER" \
    --tasks "$TASK_ARN" \
    --region "$AWS_REGION" \
    --query "tasks[0].lastStatus" \
    --output text)

  case "$STATUS" in
    RUNNING)
      echo "✅ Task is RUNNING"
      break
      ;;
    STOPPED)
      echo "❌ Task stopped before running"
      aws ecs describe-tasks \
        --profile "$AWS_PROFILE" \
        --cluster "$ECS_CLUSTER" \
        --tasks "$TASK_ARN" \
        --region "$AWS_REGION" \
        --query "tasks[0].stoppedReason" \
        --output text
      exit 1
      ;;
    *)
      echo "⏳ Current status: $STATUS"
      sleep 2
      ;;
  esac
done

# ---------- Tail logs ----------
echo "📜 Streaming logs..."
aws logs tail "$LOG_GROUP" \
  --profile "$AWS_PROFILE" \
  --log-stream-name-prefix "ecs/$CONTAINER_NAME/$TASK_ID" \
  --follow \
  --region "$AWS_REGION" &
LOG_PID=$!

# ---------- Wait for completion ----------
aws ecs wait tasks-stopped \
  --profile "$AWS_PROFILE" \
  --cluster "$ECS_CLUSTER" \
  --tasks "$TASK_ARN" \
  --region "$AWS_REGION"

# ---------- Stop log tail ----------
kill "$LOG_PID" 2>/dev/null || true

# ---------- Exit code ----------
EXIT_CODE=$(aws ecs describe-tasks \
  --profile "$AWS_PROFILE" \
  --cluster "$ECS_CLUSTER" \
  --tasks "$TASK_ARN" \
  --region "$AWS_REGION" \
  --query "tasks[0].containers[0].exitCode" \
  --output text)

if [[ "$EXIT_CODE" != "0" ]]; then
  echo "❌ Task exited with code $EXIT_CODE"
  exit 1
fi

echo "✅ Task completed successfully"
