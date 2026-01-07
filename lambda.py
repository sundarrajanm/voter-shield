import boto3
import json
import csv
import io
import re

s3 = boto3.client("s3")
ecs = boto3.client("ecs")

# ---------- CONFIG ----------
BUCKET = "264676382451-eci-download"
YEAR=2026
VERSION=1

PDF_PREFIX = f"{YEAR}/{VERSION}/S22/pdfs/"
STATE_NAME=""
AC_NAME=""

S3_OUTPUT_PATH = f"{YEAR}/{VERSION}/S22/extraction_results/"
PARTCOUNT_KEY = f"{YEAR}/{VERSION}/S22/metadata/S22_constituencies_parts_count.csv"
STATES_JSON_PATH=f"{YEAR}/{VERSION}/states.json"
CONSTITUENCY_JSON_PATH=f"{YEAR}/{VERSION}/S22/metadata/S22_constituencies.json"

ECS_CLUSTER = "default"
ECS_TASK_DEF = "voter-shield-task-1cpu-4gb"
ECS_CONTAINER = "voter-shield"
ECS_SUBNETS = ["subnet-004483a36813c238e"]
ECS_SECURITY_GROUPS = ["sg-0a47febe680b97754"]
# ----------------------------
def s3_read_json(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def get_stateNameFromCode(stateCd):
    states = s3_read_json(STATES_JSON_PATH)
    for s in states:
        if s["stateCd"] == stateCd:
            return s["stateName"]
    return None

def get_constituencyNameFromNumber(acNumber):
    constituencies = s3_read_json(CONSTITUENCY_JSON_PATH)
    for c in constituencies:
        if c["asmblyNo"] == acNumber:
            return c["asmblyName"]
    return None

def lambda_handler(event, context):
    state_cd = event.get("stateCd", "S22")
    constituency = int(event["constituency"])
    pdfs_per_batch = int(event.get("pdfs_per_batch", 1))
    circuit_breaker = int(event["circuit_breaker"])
    global STATE_NAME, AC_NAME
    STATE_NAME = get_stateNameFromCode(state_cd)
    AC_NAME = get_constituencyNameFromNumber(constituency)


    # 1️⃣ Get total parts for constituency
    total_parts = get_part_count(constituency)

    # 2️⃣ Find completed parts from result files
    completed_parts = get_completed_parts(state_cd, constituency)

    # 3️⃣ Compute remaining parts
    remaining = sorted(set(range(1, total_parts + 1)) - completed_parts)

    if not remaining:
        return {
            "status": "DONE",
            "constituency": constituency,
            "message": "All parts already processed"
        }

    # 4️⃣ Fan-out ECS tasks
    batches = create_batches(remaining, pdfs_per_batch)

    counter=0
    for batch in batches:
        trigger_ecs_task(state_cd, constituency, batch)
        counter +=1
        if(counter>= circuit_breaker):
            print("Circuit Breaker Activated..!")
            break;

    return {
        "status": "TRIGGERED",
        "constituency": constituency,
        "total_parts": total_parts,
        "completed": len(completed_parts),
        "remaining": len(remaining),
        "ecs_tasks_triggered": len(batches)
    }


# ---------- HELPERS ----------

def get_part_count(constituency):
    obj = s3.get_object(Bucket=BUCKET, Key=PARTCOUNT_KEY)
    data = obj["Body"].read().decode("utf-8")
    try:
        reader = csv.DictReader(io.StringIO(data))
        for row in reader:
            if int(row["acNumber"]) == constituency:
                part_count = int(row["totalParts"])
                print(f"Total Part Count for {constituency} is {part_count}")
                return part_count

        raise ValueError(f"Constituency {constituency} not found in partcount file")
    except Exception as e:
        print(e)

def get_completed_parts(state_cd: str, constituency: int):
    """
    Returns a SET of completed part numbers for a constituency
    by inspecting existing CSV outputs.
    """
    try:

        prefix = f"{S3_OUTPUT_PATH}{state_cd}_AC-{constituency}_"
        paginator = s3.get_paginator("list_objects_v2")

        completed_parts = set()

        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                parts = parse_parts_from_filename(obj["Key"])
                if parts:
                    completed_parts.update(parts)

        print(f"Completed parts:{completed_parts}")
        return completed_parts
    except Exception as e:
        print(e)


import re

def parse_parts_from_filename(key: str):
    """
    Extracts part numbers from filenames like:
    S22_AC-123_001-002-003-004-005_final_voter_data.csv
    """

    filename = key.split("/")[-1]

    # Match the middle part list strictly
    match = re.search(
        r"_AC-\d+_((?:\d{3}-)*\d{3})_final_voter_data\.csv$",
        filename
    )

    if not match:
        return []

    part_block = match.group(1)  # e.g. "001-002-003-004-005"
    parts = [int(p) for p in part_block.split("-")]

    return parts



def create_batches(parts, batch_size):
    return [
        parts[i:i + batch_size]
        for i in range(0, len(parts), batch_size)
    ]


def trigger_ecs_task(state_cd: str, constituency: int, batch: list[int]):
    """
    For a batch of part numbers:
    - Build exact S3 PDF URLs
    - Trigger ECS task with only those PDFs
    """

    # 1️⃣ Resolve S3 PDF URLs for this batch
    pdf_urls = []

    for part in batch:
        pdf_key = (
            f"{PDF_PREFIX}{STATE_NAME}/{AC_NAME}/"
            f"{STATE_NAME}-(S{state_cd[1:]})_"
            f"{AC_NAME}-(AC{constituency})_"
            f"{part}.pdf"
        )

        pdf_urls.append(f"s3://{BUCKET}/{pdf_key}")

    s3_input = ",".join(pdf_urls)
    print(f"S3_Input:{s3_input}")
    # 2️⃣ Build output identifier (explicit parts, not range)
    part_str = "-".join(f"{p}" for p in batch)

    output_identifier = (
        f"{state_cd}_AC-{constituency}_{part_str}"
    )
    print(f"Output Identifier:{output_identifier}")
    
    output_path= f"S3://{BUCKET}/{S3_OUTPUT_PATH}"
    print(f"Output Path:{output_path}")
    
    #:3: Fire ECS task (async)
    ecs.run_task(
        cluster=ECS_CLUSTER,
        taskDefinition=ECS_TASK_DEF,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": ECS_SUBNETS,
                "securityGroups": ECS_SECURITY_GROUPS,
                "assignPublicIp": "ENABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": ECS_CONTAINER,
                    "command": [
                        # "python",
                        # "main.py",
                        "--s3-input",
                        s3_input,
                        "--s3-output",
                        output_path,
                        "--output-identifier",
                        output_identifier,
                        # "--state-code", state_cd,
                        # "--state-name", STATE_NAME,
                        # "--ac-number", str(constituency),
                        # "--ac-name", AC_NAME,
                        # "--parts", ",".join(str(p) for p in batch),
                    ]
                }
            ]
        },
    )

