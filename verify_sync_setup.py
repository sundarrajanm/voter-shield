#!/usr/bin/env python
"""
Verification script to test S3 access and CSV file structure.

This script performs basic checks before running the full sync:
1. Tests S3 connection
2. Lists sample CSV files
3. Downloads and validates a sample CSV pair
4. Tests database connection
5. Shows what would be synced

Run this before running the full sync to catch configuration issues.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.logger import get_logger
from sync_s3_to_db import S3ToDBSyncer

logger = get_logger(__name__)


def verify_setup():
    """Verify S3 and database setup."""
    
    print("=" * 70)
    print("S3-to-DB Sync Verification")
    print("=" * 70)
    print()
    
    # Load configuration
    print("1. Loading configuration...")
    try:
        config = Config()
        print("   ✓ Configuration loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load configuration: {e}")
        return False
        
    print()
    
    # Test connections and list files
    print("2. Testing S3 connection and listing files...")
    try:
        with S3ToDBSyncer(config, dry_run=True) as syncer:
            # Default S3 location
            bucket = "264676382451-eci-download"
            prefix = "2026/1/S22/extraction_results/"
            
            print(f"   Bucket: {bucket}")
            print(f"   Prefix: {prefix}")
            print()
            
            # List CSV files
            print("3. Listing CSV files in S3...")
            csv_pairs = syncer.list_s3_csv_files(bucket, prefix)
            
            if csv_pairs:
                print(f"   ✓ Found {len(csv_pairs)} CSV pairs")
                print()
                print("   Sample files (first 5):")
                for i, pair in enumerate(csv_pairs[:5], 1):
                    print(f"   {i}. {pair.pdf_name}")
            else:
                print("   ✗ No CSV pairs found!")
                return False
                
            print()
            
            # Check database
            print("4. Checking database for existing documents...")
            existing_docs = syncer.get_existing_documents()
            print(f"   ✓ Found {len(existing_docs)} existing documents in database")
            
            if existing_docs:
                print()
                print("   Sample existing documents (first 5):")
                for i, doc in enumerate(list(existing_docs)[:5], 1):
                    print(f"   {i}. {doc}")
                    
            print()
            
            # Calculate what would be synced
            print("5. Calculating documents to sync...")
            missing_pairs = [
                pair for pair in csv_pairs 
                if pair.pdf_name not in existing_docs
            ]
            
            print(f"   ✓ Found {len(missing_pairs)} documents to sync")
            
            if missing_pairs:
                print()
                print("   Documents that would be synced (first 10):")
                for i, pair in enumerate(missing_pairs[:10], 1):
                    print(f"   {i}. {pair.pdf_name}")
                    
                if len(missing_pairs) > 10:
                    print(f"   ... and {len(missing_pairs) - 10} more")
            else:
                print("   ℹ All documents are already in the database")
                
            print()
            
            # Test downloading and parsing one file
            if missing_pairs:
                print("6. Testing CSV download and parsing (first missing doc)...")
                test_pair = missing_pairs[0]
                print(f"   Testing with: {test_pair.pdf_name}")
                
                try:
                    # Download metadata CSV
                    print(f"   Downloading metadata CSV...")
                    metadata_csv = syncer.download_csv_from_s3(bucket, test_pair.metadata_key)
                    print(f"   ✓ Downloaded ({len(metadata_csv)} bytes)")
                    
                    # Parse metadata
                    print(f"   Parsing metadata...")
                    metadata = syncer.parse_metadata_csv(metadata_csv)
                    print(f"   ✓ Parsed successfully")
                    print(f"     - pdf_name: {metadata.get('pdf_name', 'N/A')}")
                    print(f"     - state: {metadata.get('state', 'N/A')}")
                    print(f"     - district: {metadata.get('district', 'N/A')}")
                    print(f"     - total_pages: {metadata.get('total_pages', 'N/A')}")
                    print()
                    
                    # Download voters CSV
                    print(f"   Downloading voters CSV...")
                    voters_csv = syncer.download_csv_from_s3(bucket, test_pair.voters_key)
                    print(f"   ✓ Downloaded ({len(voters_csv)} bytes)")
                    
                    # Parse voters
                    print(f"   Parsing voters...")
                    voters = syncer.parse_voters_csv(voters_csv)
                    print(f"   ✓ Parsed {len(voters)} voter records")
                    
                    if voters:
                        print(f"     Sample voter (first record):")
                        v = voters[0]
                        print(f"     - serial_no: {v.get('serial_no', 'N/A')}")
                        print(f"     - epic_no: {v.get('epic_no', 'N/A')}")
                        print(f"     - name: {v.get('name', 'N/A')}")
                        print(f"     - house_no: {v.get('house_no', 'N/A')}")
                        
                    print()
                    
                except Exception as e:
                    print(f"   ✗ Failed: {e}")
                    return False
                    
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print()
    print("=" * 70)
    print("Verification Summary")
    print("=" * 70)
    print()
    print("✓ All checks passed!")
    print()
    print("Next steps:")
    print("1. Run a dry-run to preview: python sync_s3_to_db.py --dry-run --limit 5")
    print("2. Test with small batch: python sync_s3_to_db.py --limit 10")
    print("3. Run full sync: python sync_s3_to_db.py")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = verify_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nVerification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
