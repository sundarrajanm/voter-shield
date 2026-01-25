"""
S3 utilities for downloading files from AWS S3.

Supports multiple URL formats:
- s3://bucket/key
- https://bucket.s3.region.amazonaws.com/key
- https://s3.region.amazonaws.com/bucket/key
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple
from urllib.parse import urlparse, unquote

from src.logger import get_logger
from src.exceptions import ProcessingError

if TYPE_CHECKING:
    from src.config import S3Config

logger = get_logger(__name__)


# Patterns for S3 URL detection
S3_URI_PATTERN = re.compile(r"^s3://([^/]+)/(.+)$")
S3_HTTPS_VIRTUAL_HOSTED = re.compile(
    r"^https?://([^.]+)\.s3\.([^.]+\.)?amazonaws\.com/(.+)$"
)
S3_HTTPS_PATH_STYLE = re.compile(
    r"^https?://s3\.([^.]+\.)?amazonaws\.com/([^/]+)/(.+)$"
)


def is_s3_url(path: str) -> bool:
    """
    Check if a path is an S3 URL.
    
    Supports:
    - s3://bucket/key
    - https://bucket.s3.amazonaws.com/key
    - https://s3.amazonaws.com/bucket/key
    
    Args:
        path: Path or URL to check
        
    Returns:
        True if this is an S3 URL
    """
    if not isinstance(path, str):
        return False
    
    path_lower = path.lower()
    
    # S3 URI scheme
    if path_lower.startswith("s3://"):
        return True
    
    # HTTPS URLs
    if path_lower.startswith(("http://", "https://")):
        # Check for amazonaws.com
        if "amazonaws.com" in path_lower:
            return True
        # Check for custom S3-compatible endpoints
        if ".s3." in path_lower:
            return True
    
    return False


def parse_s3_url(url: str) -> Tuple[str, str]:
    """
    Parse an S3 URL into bucket and key.
    
    Args:
        url: S3 URL in various formats
        
    Returns:
        Tuple of (bucket, key)
        
    Raises:
        ValueError: If URL cannot be parsed
    """
    # s3://bucket/key
    match = S3_URI_PATTERN.match(url)
    if match:
        bucket, key = match.groups()
        return bucket, unquote(key)
    
    # https://bucket.s3.region.amazonaws.com/key
    match = S3_HTTPS_VIRTUAL_HOSTED.match(url)
    if match:
        bucket = match.group(1)
        key = match.group(3)
        return bucket, unquote(key)
    
    # https://s3.region.amazonaws.com/bucket/key
    match = S3_HTTPS_PATH_STYLE.match(url)
    if match:
        bucket = match.group(2)
        key = match.group(3)
        return bucket, unquote(key)
    
    # Try generic URL parsing as fallback
    parsed = urlparse(url)
    if parsed.scheme in ("s3", "http", "https"):
        path_parts = parsed.path.strip("/").split("/", 1)
        if len(path_parts) == 2:
            return path_parts[0], unquote(path_parts[1])
        elif parsed.netloc and path_parts:
            # Netloc is bucket for s3://
            return parsed.netloc, unquote(path_parts[0])
    
    raise ValueError(f"Cannot parse S3 URL: {url}")


def get_s3_client(s3_config: S3Config):
    """
    Create a boto3 S3 client with configuration.
    
    Args:
        s3_config: S3 configuration
        
    Returns:
        boto3 S3 client
        
    Raises:
        ImportError: If boto3 is not installed
        ProcessingError: If client creation fails
    """
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 support. Install with: pip install boto3"
        )
    
    try:
        # Build client kwargs
        kwargs = {
            "region_name": s3_config.region,
            "config": BotoConfig(
                connect_timeout=s3_config.connect_timeout,
                read_timeout=s3_config.read_timeout,
                retries={"max_attempts": s3_config.max_retries},
            ),
        }
        
        # Add explicit credentials if provided
        if s3_config.has_credentials:
            kwargs["aws_access_key_id"] = s3_config.access_key_id
            kwargs["aws_secret_access_key"] = s3_config.secret_access_key
            if s3_config.session_token:
                kwargs["aws_session_token"] = s3_config.session_token
        
        return boto3.client("s3", **kwargs)
        
    except Exception as e:
        raise ProcessingError(f"Failed to create S3 client: {e}")


def download_from_s3(
    url: str,
    s3_config: S3Config,
    local_path: Optional[Path] = None,
    download_dir: Optional[Path] = None,
) -> Path:
    """
    Download a file from S3 to local storage.
    
    Args:
        url: S3 URL (s3:// or https://)
        s3_config: S3 configuration
        local_path: Explicit local path to save to (optional)
        download_dir: Directory to download to (uses temp if not specified)
        
    Returns:
        Path to downloaded file
        
    Raises:
        ProcessingError: If download fails
        ValueError: If URL cannot be parsed
    """
    # Parse URL
    bucket, key = parse_s3_url(url)
    filename = Path(key).name
    
    logger.debug(f"Downloading from S3: bucket={bucket}, key={key}")
    
    # Determine local path
    if local_path is None:
        if download_dir:
            download_dir.mkdir(parents=True, exist_ok=True)
            local_path = download_dir / filename
        elif s3_config.download_dir:
            dl_dir = Path(s3_config.download_dir)
            dl_dir.mkdir(parents=True, exist_ok=True)
            local_path = dl_dir / filename
        else:
            # Use temp directory
            temp_dir = Path(tempfile.gettempdir()) / "electoral_roll_downloads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            local_path = temp_dir / filename
    
    # Get S3 client
    client = get_s3_client(s3_config)
    
    try:
        logger.info(f"Downloading {filename} from S3...")
        client.download_file(bucket, key, str(local_path))
        logger.info(f"Downloaded to: {local_path}")
        return local_path
        
    except Exception as e:
        raise ProcessingError(f"Failed to download from S3: {e}")


def upload_to_s3(
    local_path: Path,
    url: str,
    s3_config: S3Config,
    content_type: Optional[str] = None,
) -> str:
    """
    Upload a file to S3.
    
    Args:
        local_path: Path to local file
        url: S3 URL (s3:// or https://) or just the key
        s3_config: S3 configuration
        content_type: MIME type (optional)
        
    Returns:
        S3 URL of uploaded file
        
    Raises:
        ProcessingError: If upload fails
    """
    # Parse URL or use default bucket
    if is_s3_url(url):
        bucket, key = parse_s3_url(url)
    elif s3_config.default_bucket:
        bucket = s3_config.default_bucket
        key = url
    else:
        raise ValueError("No bucket specified and no default bucket configured")
    
    logger.debug(f"Uploading to S3: bucket={bucket}, key={key}")
    
    # Get S3 client
    client = get_s3_client(s3_config)
    
    try:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        
        client.upload_file(str(local_path), bucket, key, ExtraArgs=extra_args or None)
        
        s3_url = f"s3://{bucket}/{key}"
        logger.info(f"Uploaded to: {s3_url}")
        return s3_url
        
    except Exception as e:
        raise ProcessingError(f"Failed to upload to S3: {e}")


def check_s3_exists(url: str, s3_config: S3Config) -> bool:
    """
    Check if an object exists in S3.
    
    Args:
        url: S3 URL
        s3_config: S3 configuration
        
    Returns:
        True if object exists
    """
    try:
        bucket, key = parse_s3_url(url)
        client = get_s3_client(s3_config)
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def list_s3_objects(
    prefix: str,
    s3_config: S3Config,
    bucket: Optional[str] = None,
    max_keys: int = 1000,
) -> list:
    """
    List objects in S3 with a given prefix.
    
    Args:
        prefix: Key prefix to filter by
        s3_config: S3 configuration
        bucket: Bucket name (uses default if not specified)
        max_keys: Maximum number of keys to return
        
    Returns:
        List of object keys
    """
    bucket = bucket or s3_config.default_bucket
    if not bucket:
        raise ValueError("No bucket specified and no default bucket configured")
    
    client = get_s3_client(s3_config)
    
    try:
        response = client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=max_keys,
        )
        
        objects = response.get("Contents", [])
        return [obj["Key"] for obj in objects]
        
    except Exception as e:
        logger.error(f"Failed to list S3 objects: {e}")
        return []


def generate_presigned_url(
    url: str,
    s3_config: S3Config,
    expiration: int = 3600,
) -> str:
    """
    Generate a presigned URL for temporary access.
    
    Args:
        url: S3 URL
        s3_config: S3 configuration
        expiration: URL expiration in seconds (default: 1 hour)
        
    Returns:
        Presigned HTTPS URL
    """
    bucket, key = parse_s3_url(url)
    client = get_s3_client(s3_config)
    
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiration,
    )
