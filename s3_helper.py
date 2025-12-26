import os
from urllib.parse import urlparse

import boto3

from logger import setup_logger

logger = setup_logger()

s3 = boto3.client("s3")


def download_pdfs(pdf_s3_paths: list[str], dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)

    for s3_path in pdf_s3_paths:
        try:
            parsed = urlparse(s3_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            filename = os.path.basename(key)

            local_path = os.path.join(dest_dir, filename)
            logger.info(f"⬇️ Downloading {s3_path} → {local_path}")

            s3.download_file(bucket, key, local_path)

        except Exception as e:
            logger.error(f"❌ Failed to download {s3_path}: {e}")
            raise


def upload_directory(local_dir: str, s3_output_path: str):
    """
    Upload all files in local_dir to the given s3://bucket/prefix/
    """
    parsed = urlparse(s3_output_path)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")

    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        if not os.path.isfile(local_path):
            continue

        s3_key = f"{prefix}/{filename}"
        logger.info(f"⬆️ Uploading {local_path} → s3://{bucket}/{s3_key}")

        try:
            s3.upload_file(local_path, bucket, s3_key)
        except Exception as e:
            logger.error(f"❌ Failed to upload {filename}: {e}")
            raise
