"""
Sync CSV files from S3 to PostgreSQL database.

This script:
1. Lists CSV files in S3 bucket (metadata and voters)
2. Checks which documents are missing from the PostgreSQL metadata table
3. Downloads and imports missing CSV files to the database

Usage:
    python sync_s3_to_db.py [--dry-run] [--limit N]
    
Arguments:
    --dry-run: Show what would be synced without making changes
    --limit N: Limit the number of documents to sync (for testing)
"""

import argparse
import csv
import io
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import psycopg2
from psycopg2.extras import Json, execute_values

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.logger import get_logger
from src.utils.s3_utils import get_s3_client, list_s3_objects
from src.exceptions import ProcessingError

logger = get_logger(__name__)


@dataclass
class CSVPair:
    """Represents a pair of metadata and voters CSV files."""
    pdf_name: str  # Document name (without file extension)
    metadata_key: str  # S3 key for metadata CSV
    voters_key: str  # S3 key for voters CSV
    
    def __repr__(self):
        return f"CSVPair(pdf_name='{self.pdf_name}')"


class S3ToDBSyncer:
    """
    Syncs CSV files from S3 to PostgreSQL database.
    """
    
    def __init__(self, config: Config, dry_run: bool = False):
        """
        Initialize syncer.
        
        Args:
            config: Application configuration
            dry_run: If True, don't make any changes to the database
        """
        self.config = config
        self.dry_run = dry_run
        self.s3_client = None
        self.db_conn = None
        
    def __enter__(self):
        """Context manager entry."""
        self._connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()
        
    def _connect(self):
        """Establish S3 and database connections."""
        # S3 connection
        try:
            self.s3_client = get_s3_client(self.config.s3)
            logger.info("Connected to S3")
        except Exception as e:
            raise ProcessingError(f"Failed to connect to S3: {e}")
            
        # Database connection
        try:
            self.db_conn = psycopg2.connect(
                host=self.config.db.host,
                port=self.config.db.port,
                dbname=self.config.db.name,
                user=self.config.db.user,
                password=self.config.db.password,
                sslmode=self.config.db.ssl_mode
            )
            self.db_conn.autocommit = False
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            raise ProcessingError(f"Failed to connect to database: {e}")
            
    def _cleanup(self):
        """Clean up connections."""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Closed database connection")
            
    def list_s3_csv_files(self, bucket: str, prefix: str) -> List[CSVPair]:
        """
        List CSV files in S3 and pair metadata with voters files.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix (folder path)
            
        Returns:
            List of CSVPair objects
        """
        logger.info(f"Listing CSV files in s3://{bucket}/{prefix}")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            objects = response.get('Contents', [])
            logger.info(f"Found {len(objects)} objects in S3")
            
            # Separate metadata and voters files
            metadata_files = {}  # pdf_name -> key
            voters_files = {}    # pdf_name -> key
            
            for obj in objects:
                key = obj['Key']
                
                # Skip non-CSV files
                if not key.endswith('.csv'):
                    continue
                    
                # Extract filename from key
                filename = key.split('/')[-1]
                
                # Parse filename to get pdf_name
                # Format: "Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_metadata.csv"
                # or:     "Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_voters.csv"
                
                if filename.endswith('_metadata.csv'):
                    # Remove "_metadata.csv" to get pdf_name
                    pdf_name = filename[:-13]  # len("_metadata.csv") = 13
                    # Convert to database format: (AC118) -> (AC118_
                    # Based on user's note: CSV has (AC118)_1 but DB has (AC118_1
                    pdf_name_db = self._convert_to_db_format(pdf_name)
                    metadata_files[pdf_name_db] = key
                    
                elif filename.endswith('_voters.csv'):
                    # Remove "_voters.csv" to get pdf_name
                    pdf_name = filename[:-11]  # len("_voters.csv") = 11
                    # Convert to database format
                    pdf_name_db = self._convert_to_db_format(pdf_name)
                    voters_files[pdf_name_db] = key
                    
            # Pair them up
            csv_pairs = []
            for pdf_name, metadata_key in metadata_files.items():
                if pdf_name in voters_files:
                    csv_pairs.append(CSVPair(
                        pdf_name=pdf_name,
                        metadata_key=metadata_key,
                        voters_key=voters_files[pdf_name]
                    ))
                else:
                    logger.warning(f"No voters CSV found for: {pdf_name}")
                    
            logger.info(f"Found {len(csv_pairs)} complete CSV pairs")
            return csv_pairs
            
        except Exception as e:
            raise ProcessingError(f"Failed to list S3 objects: {e}")
            
    def _convert_to_db_format(self, csv_name: str) -> str:
        """
        Convert CSV filename format to database pdf_name format.
        
        CSV format:  Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1
        DB format:   Tamil Nadu-(S22)_Coimbatore (North)-(AC118_1
        
        The difference is the closing parenthesis before the final number.
        
        Args:
            csv_name: Name from CSV file
            
        Returns:
            Name in database format
        """
        # Pattern: ends with )_<number>
        # Replace )_<number> with _<number>
        pattern = r'\)_(\d+)$'
        match = re.search(pattern, csv_name)
        
        if match:
            # Remove the closing parenthesis before the underscore
            number = match.group(1)
            # Replace )_N with _N
            db_name = re.sub(r'\)_(\d+)$', r'_\1', csv_name)
            logger.debug(f"Converted CSV name '{csv_name}' to DB name '{db_name}'")
            return db_name
        else:
            # No match, return as-is
            logger.debug(f"No conversion needed for '{csv_name}'")
            return csv_name
            
    def get_existing_documents(self) -> set:
        """
        Get set of pdf_names that already exist in the database.
        
        Returns:
            Set of pdf_name strings
        """
        logger.info("Fetching existing documents from database...")
        
        with self.db_conn.cursor() as cur:
            cur.execute("SELECT pdf_name FROM metadata")
            results = cur.fetchall()
            
        existing = {row[0] for row in results}
        logger.info(f"Found {len(existing)} existing documents in database")
        return existing
        
    def download_csv_from_s3(self, bucket: str, key: str) -> str:
        """
        Download CSV file from S3 and return contents as string.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            CSV contents as string
        """
        logger.debug(f"Downloading s3://{bucket}/{key}")
        
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            return content
        except Exception as e:
            raise ProcessingError(f"Failed to download {key}: {e}")
            
    def parse_metadata_csv(self, csv_content: str, pdf_name_arg: str) -> Dict:
        """
        Parse metadata CSV and extract fields.
        
        Args:
            csv_content: CSV file contents as string
            pdf_name_arg: The pdf_name from the filename (used as fallback)
            
        Returns:
            Dictionary of metadata fields
        """
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)
        
        if not rows:
            raise ValueError("Metadata CSV is empty")
            
        # Metadata CSV should have 1 row
        row = rows[0]
        
        # Helper to get value from multiple possible keys
        def get_val(keys, default=None):
            if isinstance(keys, str):
                keys = [keys]
            for k in keys:
                if k in row and row[k]:
                    return row[k]
            return default

        # Use derived PDF name if missing in CSV
        pdf_name = get_val('pdf_name', pdf_name_arg)
        doc_id = get_val('document_id', pdf_name) # Use name as ID if ID missing

        # Construct JSON objects from flattened CSV fields
        # usage: get_val('administrative_address_district')
        
        # Basic fields
        data = {
            'document_id': doc_id,
            'pdf_name': pdf_name,
            'state': get_val('state', ''),
            'year': int(get_val('year', 0) or 0) or None,
            'revision_type': get_val('revision_type', ''),
            'qualifying_date': get_val('qualifying_date', ''),
            'publication_date': get_val('publication_date', ''),
            'roll_type': get_val('roll_type', ''),
            'roll_identification': get_val('roll_identification', ''),
            'total_pages': int(get_val('total_pages', 0) or 0) or None,
            'total_voters_extracted': int(get_val(['total_voters_extracted', 'total_voters'], 0) or 0) or None,
            
            # Administrative Address
            'town_or_village': get_val(['town_or_village', 'administrative_address_town_or_village'], ''),
            'main_town_or_village': get_val(['main_town_or_village', 'Main Town or Village'], ''),
            'ward_number': get_val(['ward_number', 'administrative_address_ward_number'], ''),
            'post_office': get_val(['post_office', 'administrative_address_post_office'], ''),
            'police_station': get_val(['police_station', 'administrative_address_police_station'], ''),
            'taluk_or_block': get_val(['taluk_or_block', 'administrative_address_taluk_or_block', 'Taluk or Block'], ''),
            'subdivision': get_val(['subdivision', 'administrative_address_subdivision'], ''),
            'district': get_val(['district', 'administrative_address_district'], ''),
            'pin_code': get_val(['pin_code', 'administrative_address_pin_code'], ''),
            'panchayat_name': get_val(['panchayat_name', 'Panchayat Name'], ''),
            
            # JSON structures (reconstruct if flattened)
            'constituency_details': row.get('constituency_details', '{}'),
            'administrative_address': row.get('administrative_address', '{}'),
            'polling_details': row.get('polling_details', '{}'),
            'detailed_elector_summary': row.get('detailed_elector_summary', '{}'),
            'authority_verification': row.get('authority_verification', '{}'),
            'output_identifier': row.get('output_identifier', ''),
        }
        
        # If JSON fields are empty/curled braces but we have flat data, build them?
        # For now, let's rely on what's physically in the CSV or the flat fields we mapped above
        # The main insert uses the flat fields directly for the columns.
        
        return data
        
    def parse_voters_csv(self, csv_content: str, default_document_id: str = None) -> List[Dict]:
        """
        Parse voters CSV and extract records.
        
        Args:
            csv_content: CSV file contents as string
            default_document_id: Default document ID to use if missing in CSV
            
        Returns:
            List of voter dictionaries
        """
        reader = csv.DictReader(io.StringIO(csv_content))
        voters = []
        
        for row in reader:
            # Get document_id from row or use default
            doc_id = row.get('document_id', '')
            if not doc_id and default_document_id:
                doc_id = default_document_id

            # Map CSV columns to database columns
            voter = {
                'id': row.get('id', str(uuid.uuid4())),
                'document_id': doc_id,
                'serial_no': row.get('serial_no', ''),
                'epic_no': row.get('epic_no', ''),
                'name': row.get('name', ''),
                'relation_type': row.get('relation_type', ''),
                'relation_name': row.get('relation_name', ''),
                'father_name': row.get('father_name', ''),
                'mother_name': row.get('mother_name', ''),
                'husband_name': row.get('husband_name', ''),
                'other_name': row.get('other_name', ''),
                'house_no': row.get('house_no', ''),
                'age': row.get('age', ''),
                'gender': row.get('gender', ''),
                'street_names_and_numbers': row.get('street_names_and_numbers', ''),
                'part_no': row.get('part_no', ''),
                'assembly': row.get('assembly', ''),
                'page_id': row.get('page_id', ''),
                'sequence_in_page': int(row['sequence_in_page']) if row.get('sequence_in_page') else 0,
                'epic_valid': str(row.get('epic_valid', '')).lower() in ('true', '1', 'yes'),
                'deleted': row.get('deleted', ''),
            }
            voters.append(voter)
            
        return voters
        
    def insert_metadata(self, metadata: Dict):
        """
        Insert metadata record into database.
        
        Args:
            metadata: Metadata dictionary
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would insert metadata: {metadata['pdf_name']}")
            return
            
        query = """
            INSERT INTO metadata (
                document_id, pdf_name, state, year, revision_type,
                qualifying_date, publication_date, roll_type, roll_identification,
                total_pages, total_voters_extracted, 
                town_or_village, main_town_or_village, ward_number, post_office,
                police_station, taluk_or_block, subdivision, district, pin_code, panchayat_name,
                constituency_details, administrative_address,
                polling_details, detailed_elector_summary, authority_verification,
                output_identifier
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, 
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s
            )
            ON CONFLICT (document_id) DO UPDATE SET
                pdf_name = EXCLUDED.pdf_name,
                state = EXCLUDED.state,
                year = EXCLUDED.year,
                revision_type = EXCLUDED.revision_type,
                qualifying_date = EXCLUDED.qualifying_date,
                publication_date = EXCLUDED.publication_date,
                roll_type = EXCLUDED.roll_type,
                roll_identification = EXCLUDED.roll_identification,
                total_pages = EXCLUDED.total_pages,
                total_voters_extracted = EXCLUDED.total_voters_extracted,
                town_or_village = EXCLUDED.town_or_village,
                main_town_or_village = EXCLUDED.main_town_or_village,
                ward_number = EXCLUDED.ward_number,
                post_office = EXCLUDED.post_office,
                police_station = EXCLUDED.police_station,
                taluk_or_block = EXCLUDED.taluk_or_block,
                subdivision = EXCLUDED.subdivision,
                district = EXCLUDED.district,
                pin_code = EXCLUDED.pin_code,
                panchayat_name = EXCLUDED.panchayat_name,
                constituency_details = EXCLUDED.constituency_details,
                administrative_address = EXCLUDED.administrative_address,
                polling_details = EXCLUDED.polling_details,
                detailed_elector_summary = EXCLUDED.detailed_elector_summary,
                authority_verification = EXCLUDED.authority_verification,
                output_identifier = EXCLUDED.output_identifier
        """
        
        # Parse JSON fields (they might be strings in CSV)
        import json
        
        def parse_json_field(field_value):
            if isinstance(field_value, str):
                try:
                    return json.loads(field_value)
                except:
                    return {}
            return field_value or {}
            
        with self.db_conn.cursor() as cur:
            cur.execute(query, (
                metadata['document_id'],
                metadata['pdf_name'],
                metadata['state'],
                metadata['year'],
                metadata['revision_type'],
                metadata['qualifying_date'],
                metadata['publication_date'],
                metadata['roll_type'],
                metadata['roll_identification'],
                metadata['total_pages'],
                metadata['total_voters_extracted'],
                metadata['town_or_village'],
                metadata['main_town_or_village'],
                metadata['ward_number'],
                metadata['post_office'],
                metadata['police_station'],
                metadata['taluk_or_block'],
                metadata['subdivision'],
                metadata['district'],
                metadata['pin_code'],
                metadata['panchayat_name'],
                Json(parse_json_field(metadata['constituency_details'])),
                Json(parse_json_field(metadata['administrative_address'])),
                Json(parse_json_field(metadata['polling_details'])),
                Json(parse_json_field(metadata['detailed_elector_summary'])),
                Json(parse_json_field(metadata['authority_verification'])),
                metadata['output_identifier']
            ))
            
        self.db_conn.commit()
        logger.info(f"Inserted metadata: {metadata['pdf_name']}")
        
    def insert_voters(self, voters: List[Dict]):
        """
        Insert voter records into database.
        
        Args:
            voters: List of voter dictionaries
        """
        if not voters:
            return
            
        if self.dry_run:
            logger.info(f"[DRY RUN] Would insert {len(voters)} voters")
            return
        
        # Get document_id from first voter (all voters should have same document_id)
        document_id = voters[0].get('document_id') if voters else None
        if not document_id:
            logger.error("Cannot insert voters: no document_id found")
            return
            
        # Delete existing voters for this document to prevent duplicates
        delete_query = "DELETE FROM voters WHERE document_id = %s"
        with self.db_conn.cursor() as cur:
            cur.execute(delete_query, (document_id,))
            deleted_count = cur.rowcount
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} existing voters for document {document_id}")
            
        query = """
            INSERT INTO voters (
                id, document_id, serial_no, epic_no, name,
                relation_type, relation_name, 
                father_name, mother_name, husband_name, other_name,
                house_no, age, gender, street_names_and_numbers, part_no, assembly,
                page_id, sequence_in_page, epic_valid, deleted
            ) VALUES %s
        """
        
        values = []
        for voter in voters:
            values.append((
                voter['id'],
                voter['document_id'],
                voter['serial_no'],
                voter['epic_no'],
                voter['name'],
                voter['relation_type'],
                voter['relation_name'],
                voter['father_name'],
                voter['mother_name'],
                voter['husband_name'],
                voter['other_name'],
                voter['house_no'],
                voter['age'],
                voter['gender'],
                voter['street_names_and_numbers'],
                voter['part_no'],
                voter['assembly'],
                voter['page_id'],
                voter['sequence_in_page'],
                voter['epic_valid'],
                voter['deleted']
            ))
            
        with self.db_conn.cursor() as cur:
            execute_values(cur, query, values)
            
        self.db_conn.commit()
        logger.info(f"Inserted {len(voters)} voters for document {document_id}")
        
        
    def _ensure_connection(self):
        """Ensure database connection is active, reconnect if needed."""
        if self.db_conn is None or self.db_conn.closed:
            logger.info("Database connection closed or missing. Reconnecting...")
            self._connect()
            return

        # Optional: Active check
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("SELECT 1")
        except psycopg2.OperationalError:
            logger.info("Database connection dead. Reconnecting...")
            self._connect()

    def sync_document(self, csv_pair: CSVPair, bucket: str):
        """
        Sync a single document (metadata + voters) from S3 to database.
        
        Args:
            csv_pair: CSV pair to sync
            bucket: S3 bucket name
        """
        logger.info(f"Syncing document: {csv_pair.pdf_name}")
        
        # Ensure connection before starting a transaction
        self._ensure_connection()
        
        try:
            # Download and parse metadata CSV
            metadata_csv = self.download_csv_from_s3(bucket, csv_pair.metadata_key)
            metadata = self.parse_metadata_csv(metadata_csv, csv_pair.pdf_name)
            
            # Download and parse voters CSV
            voters_csv = self.download_csv_from_s3(bucket, csv_pair.voters_key)
            voters = self.parse_voters_csv(voters_csv, default_document_id=metadata['document_id'])
            
            # Insert into database
            self.insert_metadata(metadata)
            self.insert_voters(voters)
            
            logger.info(f"Successfully synced: {csv_pair.pdf_name}")
            
        except psycopg2.OperationalError as e:
            # If connection drops during processing, we can't easily retry mid-transaction
            # But we can raise it so the main loop can reconnect and retry this document
            logger.warning(f"Database connection lost during {csv_pair.pdf_name}: {e}")
            if self.db_conn:
                try:
                    self.db_conn.close()
                except:
                    pass
            self.db_conn = None 
            raise # Re-raise to trigger main loop retry logic
            
        except Exception as e:
            logger.error(f"Failed to sync {csv_pair.pdf_name}: {e}", exc_info=True)
            if self.db_conn and not self.db_conn.closed:
                self.db_conn.rollback()
                
    def sync_all(self, bucket: str, prefix: str, limit: Optional[int] = None):
        """
        Sync all missing documents from S3 to database.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix (folder path)
            limit: Optional limit on number of documents to sync
        """
        # List CSV files in S3
        csv_pairs = self.list_s3_csv_files(bucket, prefix)
        
        # Get existing documents from database
        self._ensure_connection()
        existing_docs = self.get_existing_documents()
        
        # Filter out documents that already exist
        missing_pairs = [
            pair for pair in csv_pairs 
            if pair.pdf_name not in existing_docs
        ]
        
        logger.info(f"Found {len(missing_pairs)} missing documents to sync")
        
        if limit:
            missing_pairs = missing_pairs[:limit]
            logger.info(f"Limiting to {len(missing_pairs)} documents")
            
        # Sync each missing document
        for i, csv_pair in enumerate(missing_pairs, 1):
            logger.info(f"[{i}/{len(missing_pairs)}] Processing: {csv_pair.pdf_name}")
            
            # Retry loop for single document
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.sync_document(csv_pair, bucket)
                    break # Success, move to next document
                except psycopg2.OperationalError:
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying connection for {csv_pair.pdf_name} (Attempt {attempt+2}/{max_retries})")
                        import time
                        time.sleep(5) # Wait before retry
                        continue
                    else:
                        logger.error(f"Failed to sync {csv_pair.pdf_name} after {max_retries} connection attempts.")
                except Exception:
                    break # Other errors handled inside sync_document, move to next
            
        logger.info(f"Sync complete! Processed {len(missing_pairs)} documents")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sync CSV files from S3 to PostgreSQL database"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be synced without making changes"
    )
    parser.add_argument(
        '--limit',
        type=int,
        help="Limit number of documents to sync (for testing)"
    )
    parser.add_argument(
        '--bucket',
        default='264676382451-eci-download',
        help="S3 bucket name (default: 264676382451-eci-download)"
    )
    parser.add_argument(
        '--prefix',
        default='2026/1/S22/extraction_results/',
        help="S3 prefix/folder path (default: 2026/1/S22/extraction_results/)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
        
    # Run sync
    try:
        with S3ToDBSyncer(config, dry_run=args.dry_run) as syncer:
            syncer.sync_all(
                bucket=args.bucket,
                prefix=args.prefix,
                limit=args.limit
            )
            
        logger.info("Sync completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Sync interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
