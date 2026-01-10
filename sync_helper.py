#!/usr/bin/env python
"""
Quick reference script for S3-to-DB sync operations.

This shows common usage patterns for the sync_s3_to_db.py script.
"""

import os
import sys

def show_usage():
    """Display common usage patterns."""
    
    print("=" * 70)
    print("S3 to Database Sync - Common Usage Patterns")
    print("=" * 70)
    print()
    
    print("1. DRY RUN - Preview what would be synced (RECOMMENDED FIRST STEP):")
    print("   python sync_s3_to_db.py --dry-run")
    print()
    
    print("2. DRY RUN with limit (test with 5 documents):")
    print("   python sync_s3_to_db.py --dry-run --limit 5")
    print()
    
    print("3. SYNC LIMITED BATCH (sync first 10 documents):")
    print("   python sync_s3_to_db.py --limit 10")
    print()
    
    print("4. FULL SYNC (sync all missing documents):")
    print("   python sync_s3_to_db.py")
    print()
    
    print("5. Custom S3 location:")
    print("   python sync_s3_to_db.py --bucket my-bucket --prefix my/path/")
    print()
    
    print("=" * 70)
    print("Configuration Check")
    print("=" * 70)
    print()
    
    # Check if .env exists
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        print("✓ .env file found")
        
        # Check for required variables
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
            'DB_HOST',
            'DB_NAME',
            'DB_USER',
            'DB_PASSWORD'
        ]
        
        missing = []
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)
        
        if missing:
            print("✗ Missing environment variables:")
            for var in missing:
                print(f"  - {var}")
        else:
            print("✓ All required environment variables are set")
    else:
        print("✗ .env file not found")
        print("  Create .env from .env.example and configure it")
    
    print()
    print("=" * 70)
    print("Default S3 Location")
    print("=" * 70)
    print()
    print("Bucket: 264676382451-eci-download2026")
    print("Prefix: 1/S22/extraction_results/")
    print()
    print("CSV File Format:")
    print("  - Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_metadata.csv")
    print("  - Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_voters.csv")
    print()
    print("Database pdf_name Format:")
    print("  - Tamil Nadu-(S22)_Coimbatore (North)-(AC118_1")
    print()
    print("=" * 70)
    print()

if __name__ == "__main__":
    show_usage()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        print("Running dry-run preview...")
        print()
        os.system("python sync_s3_to_db.py --dry-run --limit 5")
