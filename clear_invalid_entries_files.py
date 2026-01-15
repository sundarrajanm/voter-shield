"""
Script to delete metadata and voters CSV files from S3 and database based on database validation
Includes progress tracking, resumability, and detailed logging.
"""

import os
import json
import re
import argparse
import boto3
import psycopg2
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime

# Load environment variables
load_dotenv()

# Constants
STATUS_FILE = "deletion_status.json"
OUTPUT_FILE = "invalid.json"
OUTPUT_DIR = "output"
S3_BUCKET = "264676382451-eci-download"
S3_BASE_PATH = "2026/1/S22/extraction_results"

class DatabaseS3Cleaner:
    def __init__(self, s3_output_path=None):
        """Initialize S3 client, database connection and load configuration"""
        # S3 Client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'ap-south-1')
        )
        
        # Database Connection
        self.db_conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT', 5432),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        self.db_conn.autocommit = False  # Use transactions
        
        self.bucket = S3_BUCKET
        self.base_path = S3_BASE_PATH
        self.s3_output_path = s3_output_path
        self.status = self.load_status()
        
        # Tracking
        self.removed_booths = []
        self.rerun_assemblies = {}  # Dict to track unique assemblies
        
        # Counters
        self.s3_metadata_deleted = 0
        self.s3_voters_deleted = 0
        self.s3_metadata_failed = 0
        self.s3_voters_failed = 0
        self.s3_metadata_not_found = 0
        self.s3_voters_not_found = 0
        self.db_voters_deleted = 0
        self.db_metadata_deleted = 0
        self.db_failed = 0
        
    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db_conn') and self.db_conn:
            self.db_conn.close()
    
    def load_status(self):
        """Load status from JSON file for resumability"""
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        return {
            "completed": [],
            "failed": [],
            "last_updated": None,
            "started_at": datetime.now().isoformat()
        }
    
    def save_status(self):
        """Save current status to JSON file"""
        self.status["last_updated"] = datetime.now().isoformat()
        with open(STATUS_FILE, 'w') as f:
            json.dump(self.status, indent=2, fp=f)
    
    def get_invalid_records(self):
        """Query database for invalid metadata records"""
        query = """
            SELECT document_id
            FROM metadata
            WHERE total <> total_voters_extracted;
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(query)
        records = cursor.fetchall()
        cursor.close()
        
        # Extract document_ids
        document_ids = [record[0] for record in records]
        return document_ids
    
    def parse_assembly_info(self, document_id):
        """
        Parse assembly name and ID from document_id
        Format: Tamil Nadu-(S22)_Kavundampalayam-(AC117)_319
        Returns: (assembly_name, assembly_id) or (None, None) if parsing fails
        """
        try:
            # Pattern to extract assembly name and ID
            # Format: {state}_{assembly_name}-(AC{id})_{booth_number}
            pattern = r'^[^_]+_([^-]+)-\(AC(\d+)\)_\d+$'
            match = re.match(pattern, document_id)
            
            if match:
                assembly_name = match.group(1)
                assembly_id = match.group(2)
                return assembly_name, assembly_id
            else:
                print(f"Warning: Could not parse assembly info from: {document_id}")
                return None, None
        except Exception as e:
            print(f"Error parsing document_id '{document_id}': {e}")
            return None, None
    
    def delete_s3_file(self, s3_key):
        """
        Delete a single file from S3
        Returns: ('success', None) or ('not_found', None) or ('error', error_message)
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            return ('success', None)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404' or error_code == 'NoSuchKey':
                return ('not_found', None)
            return ('error', str(e))
        except Exception as e:
            return ('error', str(e))
    
    def delete_s3_files(self, document_id):
        """
        Delete both metadata and voters CSV files from S3 for a given document_id
        Returns: dict with deletion results
        """
        # Extract constituency folder from filename
        # Format: "Tamil Nadu-(S22)_Sivaganga-(AC186)_329"
        parts = document_id.split('_')
        if len(parts) >= 2:
            constituency_folder = parts[1]
        else:
            constituency_folder = ""
            print(f"Warning: Unexpected document_id format: {document_id}")
        
        # Construct S3 keys
        if constituency_folder:
            metadata_key = f"{self.base_path}/{constituency_folder}/{document_id}_metadata.csv"
            voters_key = f"{self.base_path}/{constituency_folder}/{document_id}_voters.csv"
        else:
            metadata_key = f"{self.base_path}/{document_id}_metadata.csv"
            voters_key = f"{self.base_path}/{document_id}_voters.csv"
        
        result = {
            "document_id": document_id,
            "metadata": {},
            "voters": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Delete metadata file
        status, error = self.delete_s3_file(metadata_key)
        result["metadata"]["status"] = status
        result["metadata"]["s3_key"] = metadata_key
        if error:
            result["metadata"]["error"] = error
        
        if status == 'success':
            self.s3_metadata_deleted += 1
        elif status == 'not_found':
            self.s3_metadata_not_found += 1
        else:
            self.s3_metadata_failed += 1
        
        # Delete voters file
        status, error = self.delete_s3_file(voters_key)
        result["voters"]["status"] = status
        result["voters"]["s3_key"] = voters_key
        if error:
            result["voters"]["error"] = error
        
        if status == 'success':
            self.s3_voters_deleted += 1
        elif status == 'not_found':
            self.s3_voters_not_found += 1
        else:
            self.s3_voters_failed += 1
        
        return result
    
    def delete_db_records(self, document_id):
        """
        Delete records from voters and metadata tables for given document_id
        Returns: dict with deletion results
        """
        result = {
            "voters_deleted": 0,
            "metadata_deleted": 0,
            "error": None
        }
        
        try:
            cursor = self.db_conn.cursor()
            
            # Delete from voters table
            cursor.execute("DELETE FROM voters WHERE document_id = %s", (document_id,))
            result["voters_deleted"] = cursor.rowcount
            self.db_voters_deleted += cursor.rowcount
            
            # Delete from metadata table
            cursor.execute("DELETE FROM metadata WHERE document_id = %s", (document_id,))
            result["metadata_deleted"] = cursor.rowcount
            self.db_metadata_deleted += cursor.rowcount
            
            # Commit transaction
            self.db_conn.commit()
            cursor.close()
            
        except Exception as e:
            self.db_conn.rollback()
            result["error"] = str(e)
            self.db_failed += 1
            print(f"Database deletion error for {document_id}: {e}")
        
        return result
    
    def process_document(self, document_id):
        """
        Process a single document: delete S3 files and database records
        Returns: dict with all results
        """
        result = {
            "document_id": document_id,
            "s3": {},
            "database": {},
            "assembly": {}
        }
        
        # Delete S3 files
        result["s3"] = self.delete_s3_files(document_id)
        
        # Delete database records
        result["database"] = self.delete_db_records(document_id)
        
        # Parse assembly info
        assembly_name, assembly_id = self.parse_assembly_info(document_id)
        if assembly_name and assembly_id:
            result["assembly"]["name"] = assembly_name
            result["assembly"]["id"] = assembly_id
            
            # Track unique assemblies
            if assembly_id not in self.rerun_assemblies:
                self.rerun_assemblies[assembly_id] = {
                    "assembly_name": assembly_name,
                    "assembly_id": assembly_id
                }
        
        # Add to removed booths
        self.removed_booths.append(document_id)
        
        return result
    
    def generate_output_file(self):
        """Generate invalid.json with removed_booths and rerun_assemblies"""
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        output_data = {
            "removed_booths": self.removed_booths,
            "rerun_assemblies": list(self.rerun_assemblies.values())
        }
        
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"\n✓ Output file generated: {output_path}")
        
        # Upload to S3 if path provided
        if self.s3_output_path:
            self.upload_to_s3(output_path)
        
        return output_path
    
    def upload_to_s3(self, local_file_path):
        """Upload invalid.json to S3"""
        try:
            # Parse S3 path
            # Expected format: s3://bucket/path/to/folder/ or bucket/path/to/folder/
            s3_path = self.s3_output_path.replace('s3://', '')
            s3_path = s3_path.rstrip('/')
            
            # Split bucket and key
            parts = s3_path.split('/', 1)
            if len(parts) == 2:
                bucket, prefix = parts
                s3_key = f"{prefix}/{OUTPUT_FILE}"
            else:
                bucket = parts[0]
                s3_key = OUTPUT_FILE
            
            # Upload file
            self.s3_client.upload_file(local_file_path, bucket, s3_key)
            print(f"✓ Uploaded to S3: s3://{bucket}/{s3_key}")
            
        except Exception as e:
            print(f"⚠ Failed to upload to S3: {e}")
    
    def process_all(self):
        """Main processing logic"""
        print("=" * 80)
        print("Database-Driven S3 and Database Cleanup Script")
        print("=" * 80)
        print(f"S3 Bucket: {self.bucket}")
        print(f"S3 Base Path: {self.base_path}")
        print(f"Status File: {STATUS_FILE}")
        if self.s3_output_path:
            print(f"S3 Output Path: {self.s3_output_path}")
        print("=" * 80)
        
        # Get invalid records from database
        print("\nQuerying database for invalid records...")
        all_document_ids = self.get_invalid_records()
        print(f"Found {len(all_document_ids)} invalid records in database")
        
        if not all_document_ids:
            print("\n✓ No invalid records found!")
            return
        
        # Filter out already completed
        completed_ids = set(self.status.get("completed", []))
        ids_to_process = [doc_id for doc_id in all_document_ids if doc_id not in completed_ids]
        
        if completed_ids:
            print(f"Already completed: {len(completed_ids)}")
            print(f"Remaining to process: {len(ids_to_process)}")
        
        if not ids_to_process:
            print("\n✓ All records already processed!")
            return
        
        print("\nStarting cleanup process...\n")
        
        # Process each document with progress bar
        with tqdm(total=len(ids_to_process), desc="Processing documents", unit="doc") as pbar:
            for document_id in ids_to_process:
                # Process document
                result = self.process_document(document_id)
                
                # Update status
                has_errors = (
                    result["s3"]["metadata"]["status"] == 'error' or
                    result["s3"]["voters"]["status"] == 'error' or
                    result["database"].get("error") is not None
                )
                
                if not has_errors:
                    self.status["completed"].append(document_id)
                else:
                    self.status["failed"].append({
                        "document_id": document_id,
                        "result": result
                    })
                
                # Save status periodically (every 10 documents)
                if len(self.status["completed"]) % 10 == 0:
                    self.save_status()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'S3_del': self.s3_metadata_deleted + self.s3_voters_deleted,
                    'DB_del': self.db_voters_deleted + self.db_metadata_deleted,
                    'Fails': len(self.status["failed"])
                })
        
        # Final save
        self.save_status()
        
        # Generate output file
        self.generate_output_file()
        
        # Print summary
        self.print_summary(len(all_document_ids))
    
    def print_summary(self, total_documents):
        """Print cleanup summary"""
        print("\n" + "=" * 80)
        print("CLEANUP SUMMARY")
        print("=" * 80)
        print(f"\nTotal documents processed: {len(self.status['completed'])}/{total_documents}")
        
        print(f"\nS3 Metadata Files:")
        print(f"  ✓ Successfully deleted: {self.s3_metadata_deleted}")
        print(f"  ⚠ Not found: {self.s3_metadata_not_found}")
        print(f"  ✗ Failed to delete: {self.s3_metadata_failed}")
        
        print(f"\nS3 Voters Files:")
        print(f"  ✓ Successfully deleted: {self.s3_voters_deleted}")
        print(f"  ⚠ Not found: {self.s3_voters_not_found}")
        print(f"  ✗ Failed to delete: {self.s3_voters_failed}")
        
        print(f"\nDatabase Operations:")
        print(f"  ✓ Voter records deleted: {self.db_voters_deleted}")
        print(f"  ✓ Metadata records deleted: {self.db_metadata_deleted}")
        print(f"  ✗ Failed operations: {self.db_failed}")
        
        print(f"\nOutput:")
        print(f"  ✓ Removed booths: {len(self.removed_booths)}")
        print(f"  ✓ Assemblies to rerun: {len(self.rerun_assemblies)}")
        
        if self.status.get("failed"):
            print(f"\n⚠ Documents with errors: {len(self.status['failed'])}")
            print(f"  Check {STATUS_FILE} for details")
        
        print("\n" + "=" * 80)
        print(f"Status saved to: {STATUS_FILE}")
        print("=" * 80)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Delete invalid metadata and voters from S3 and database'
    )
    parser.add_argument(
        '--s3-output',
        type=str,
        help='S3 folder path to upload invalid.json (e.g., s3://bucket/path/to/folder/)'
    )
    
    args = parser.parse_args()
    
    try:
        cleaner = DatabaseS3Cleaner(s3_output_path=args.s3_output)
        cleaner.process_all()
    except psycopg2.Error as e:
        print(f"Database Error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
