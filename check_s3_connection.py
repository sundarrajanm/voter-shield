#!/usr/bin/env python
"""
Check S3 connection and list available buckets.

This script helps verify your S3 configuration and access.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.logger import get_logger
from src.utils.s3_utils import get_s3_client

logger = get_logger(__name__)


def check_s3_connection():
    """Check S3 connection and list buckets."""
    
    print("=" * 70)
    print("S3 Connection Check")
    print("=" * 70)
    print()
    
    # Load configuration
    print("1. Loading configuration...")
    try:
        config = Config()
        print("   ✓ Configuration loaded")
        print(f"   AWS Region: {config.s3.region}")
        print(f"   Has Credentials: {config.s3.has_credentials}")
        if config.s3.default_bucket:
            print(f"   Default Bucket: {config.s3.default_bucket}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
        
    print()
    
    # Test S3 connection
    print("2. Connecting to S3...")
    try:
        s3_client = get_s3_client(config.s3)
        print("   ✓ S3 client created")
    except Exception as e:
        print(f"   ✗ Failed to create S3 client: {e}")
        print()
        print("Possible issues:")
        print("- AWS credentials not configured")
        print("- boto3 not installed")
        return False
        
    print()
    
    # List buckets
    print("3. Listing available buckets...")
    try:
        response = s3_client.list_buckets()
        buckets = response.get('Buckets', [])
        
        if buckets:
            print(f"   ✓ Found {len(buckets)} buckets:")
            for bucket in buckets:
                print(f"   - {bucket['Name']}")
        else:
            print("   ℹ No buckets found")
            
    except Exception as e:
        print(f"   ✗ Failed to list buckets: {e}")
        print()
        print("Possible issues:")
        print("- Invalid AWS credentials")
        print("- Insufficient permissions")
        print("- Network connectivity issues")
        return False
        
    print()
    
    # Try the default bucket
    target_bucket = "264676382451-eci-download"
    print(f"4. Checking target bucket: {target_bucket}")
    try:
        # Try to head the bucket
        s3_client.head_bucket(Bucket=target_bucket)
        print(f"   ✓ Bucket exists and is accessible")
        
        # Try to list a few objects
        print(f"   Listing sample objects...")
        response = s3_client.list_objects_v2(
            Bucket=target_bucket,
            Prefix="2026/1/S22/extraction_results/",
            MaxKeys=10
        )
        
        objects = response.get('Contents', [])
        if objects:
            print(f"   ✓ Found {len(objects)} objects (showing first 10):")
            for obj in objects[:10]:
                print(f"   - {obj['Key']}")
        else:
            print(f"   ℹ No objects found with prefix '2026/1/S22/extraction_results/'")
            
    except Exception as e:
        error_str = str(e)
        print(f"   ✗ Error: {error_str}")
        print()
        
        if "NoSuchBucket" in error_str:
            print("The bucket does not exist or you don't have access to it.")
            print()
            print("Please verify:")
            print("1. The bucket name is correct")
            print("2. Your AWS credentials have access to this bucket")
            print("3. The bucket is in the correct region")
            print()
            print("Available buckets (from step 3 above):")
            if buckets:
                for bucket in buckets:
                    print(f"   - {bucket['Name']}")
            else:
                print("   (no buckets accessible with current credentials)")
        elif "AccessDenied" in error_str or "Forbidden" in error_str:
            print("Access denied to the bucket.")
            print("Your credentials don't have permission to access this bucket.")
        else:
            print("Unknown error occurred.")
            
        return False
        
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("✓ S3 connection successful!")
    print(f"✓ Target bucket '{target_bucket}' is accessible")
    print()
    print("You can now run the sync script:")
    print("  python sync_s3_to_db.py --dry-run --limit 5")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = check_s3_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
