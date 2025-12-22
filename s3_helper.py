import boto3
from urllib.parse import urlparse

s3 = boto3.client("s3")

def download_pdfs(pdf_s3_paths, dest_dir):
    for s3_path in pdf_s3_paths:
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        filename = os.path.basename(key)
        local_path = os.path.join(dest_dir, filename)
        s3.download_file(bucket, key, local_path)
def upload_result():
    pass