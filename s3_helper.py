import boto3
import os
from urllib.parse import urlparse
from logger import setup_logger
logger = setup_logger()

s3 = boto3.client("s3")

def download_pdfs(pdf_s3_paths, dest_dir):
    try:
        for s3_path in pdf_s3_paths:
            parsed = urlparse(s3_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            filename = os.path.basename(key)
            local_path = os.path.join(dest_dir, filename)
            s3.download_file(bucket, key, local_path)
    except Exception as e:
        logger.error(f"Error while downloading this file:{s3_path}: Exception:{e}")
        raise e
def upload_result():
    pass
