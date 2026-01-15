"""
Sync CSV files from S3 to PostgreSQL database.

This script:
1. Discovers assembly sub-folders within the S3 prefix (one level deep)
2. Lists CSV files in each sub-folder (metadata and voters)
3. Checks database by pdf_name to skip existing documents (no download needed)
4. For new documents: Downloads CSVs from S3 (or uses cache if available)
5. Inserts only NEW documents to the database (never updates existing ones)
6. Generates a timestamped JSON report of all operations

Caching Strategy:
- Dry-run (--dry-run): Downloads new files and saves to csv_cache/ folder
- Normal run: Uses cached files if available, then auto-deletes cache after completion

Usage:
    # Preview sync and cache files for later
    python sync_s3_to_db.py --dry-run
    
    # Actually sync using cached files (if available from dry-run)
    python sync_s3_to_db.py
    
Arguments:
    --dry-run: Show what would be synced without making changes (saves cache)
    --limit N: Limit the number of documents to sync (for testing)
    --s3-output: S3 path to upload the sync report JSON
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal
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
    assembly_folder: str = ""  # Assembly folder name
    
    def __repr__(self):
        return f"CSVPair(pdf_name='{self.pdf_name}', assembly='{self.assembly_folder}')"


@dataclass
class SyncOperation:
    """Represents a single sync operation."""
    document_id: str
    pdf_name: str
    operation: Literal['created', 'updated', 'skipped']
    assembly_folder: str
    timestamp: str
    voters_count: int = 0
    

@dataclass
class SyncReport:
    """Aggregate sync report."""
    sync_timestamp: str
    total_documents_processed: int
    documents_created: int
    documents_updated: int
    documents_skipped: int
    assemblies_processed: List[str]
    operations: List[Dict]


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
        self.cache_dir = Path("csv_cache")
        
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
            
            # Test the connection
            with self.db_conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if not result or result[0] != 1:
                    raise Exception("Database connection test failed")
            
            logger.info("Connected to PostgreSQL database (connection verified)")
        except Exception as e:
            if self.config.db.required:
                raise ProcessingError(f"Failed to connect to database: {e}")
            else:
                logger.warning(f"Failed to connect to database: {e}")
                logger.warning("DB_REQUIRED=false, continuing without database (sync operations will be limited)")
                self.db_conn = None
            
    def _cleanup(self):
        """Clean up connections."""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Closed database connection")
            
    def list_s3_csv_files(self, bucket: str, prefix: str) -> List[CSVPair]:
        """
        List CSV files in S3 by discovering sub-folders and pairing metadata with voters files.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix (folder path)
            
        Returns:
            List of CSVPair objects
        """
        logger.info(f"Discovering assembly sub-folders in s3://{bucket}/{prefix}")
        
        try:
            # First, list all objects with the prefix to discover sub-folders
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                Delimiter='/'
            )
            
            # Get sub-folders (CommonPrefixes)
            common_prefixes = response.get('CommonPrefixes', [])
            assembly_folders = [cp['Prefix'] for cp in common_prefixes]
            
            logger.info(f"Found {len(assembly_folders)} assembly sub-folders")
            
            all_csv_pairs = []
            
            # Process each assembly folder
            for assembly_prefix in assembly_folders:
                assembly_name = assembly_prefix.rstrip('/').split('/')[-1]
                logger.info(f"Processing assembly folder: {assembly_name}")
                
                # List CSV files in this assembly folder
                folder_response = self.s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=assembly_prefix
                )
                
                objects = folder_response.get('Contents', [])
                logger.info(f"  Found {len(objects)} objects in {assembly_name}")
                
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
                for pdf_name, metadata_key in metadata_files.items():
                    if pdf_name in voters_files:
                        all_csv_pairs.append(CSVPair(
                            pdf_name=pdf_name,
                            metadata_key=metadata_key,
                            voters_key=voters_files[pdf_name],
                            assembly_folder=assembly_name
                        ))
                    else:
                        logger.warning(f"  No voters CSV found for: {pdf_name} in {assembly_name}")
                        
                logger.info(f"  Found {len([p for p in all_csv_pairs if p.assembly_folder == assembly_name])} complete CSV pairs in {assembly_name}")
                    
            logger.info(f"Total: Found {len(all_csv_pairs)} complete CSV pairs across all assemblies")
            return all_csv_pairs
            
        except Exception as e:
            raise ProcessingError(f"Failed to list S3 objects: {e}")
            
    def _convert_to_db_format(self, csv_name: str) -> str:
        """
        Return CSV filename as-is (no conversion needed).
        
        Args:
            csv_name: Name from CSV file
            
        Returns:
            Name in database format (same as input)
        """
        # No conversion needed - use filename as-is
        return csv_name
            
    def get_existing_documents(self) -> set:
        """
        Get set of pdf_names that already exist in the database.
        
        Returns:
            Set of pdf_name strings (empty set if no database connection)
        """
        if self.db_conn is None:
            logger.warning("No database connection available, returning empty set of existing documents")
            return set()
            
        logger.info("Fetching existing documents from database...")
        
        with self.db_conn.cursor() as cur:
            cur.execute("SELECT pdf_name FROM metadata")
            results = cur.fetchall()
            
        existing = {row[0] for row in results}
        logger.info(f"Found {len(existing)} existing documents in database")
        return existing
        
    def download_csv_from_s3(self, bucket: str, key: str, cache_key: Optional[str] = None) -> str:
        """
        Download CSV file from S3 and return contents as string.
        Uses cache if available, saves to cache during dry-run.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            cache_key: Optional cache file path (relative to cache_dir)
            
        Returns:
            CSV contents as string
        """
        # Check cache first if cache_key provided
        if cache_key:
            cache_file = self.cache_dir / cache_key
            if cache_file.exists():
                logger.debug(f"Using cached file: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
        
        # Download from S3
        logger.debug(f"Downloading s3://{bucket}/{key}")
        
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            
            # Save to cache if cache_key provided and in dry-run mode
            if cache_key and self.dry_run:
                cache_file = self.cache_dir / cache_key
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"Cached file: {cache_file}")
            
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
            
            # JSON structures - build from flattened CSV fields
            'constituency_details': {
                'assembly_constituency_name': get_val('constituency_details_assembly_constituency_name', ''),
                'assembly_constituency_number': get_val('constituency_details_assembly_constituency_number', ''),
                'assembly_reservation_status': get_val('constituency_details_assembly_reservation_status', ''),
                'parliamentary_constituency_name': get_val('constituency_details_parliamentary_constituency_name', ''),
                'parliamentary_constituency_number': get_val('constituency_details_parliamentary_constituency_number', ''),
                'parliamentary_reservation_status': get_val('constituency_details_parliamentary_reservation_status', ''),
                'part_number': get_val('constituency_details_part_number', ''),
            },
            'administrative_address': {
                'town_or_village': get_val(['town_or_village', 'administrative_address_town_or_village'], ''),
                'main_town_or_village': get_val(['main_town_or_village', 'Main Town or Village'], ''),
                'ward_number': get_val(['ward_number', 'administrative_address_ward_number'], ''),
                'post_office': get_val(['post_office', 'administrative_address_post_office', 'Post Office'], ''),
                'police_station': get_val(['police_station', 'administrative_address_police_station'], ''),
                'taluk_or_block': get_val(['taluk_or_block', 'administrative_address_taluk_or_block', 'Taluk or Block'], ''),
                'subdivision': get_val(['subdivision', 'administrative_address_subdivision', 'Subdivision'], ''),
                'district': get_val(['district', 'administrative_address_district'], ''),
                'pin_code': get_val(['pin_code', 'administrative_address_pin_code'], ''),
                'panchayat_name': get_val(['panchayat_name', 'Panchayat Name'], ''),
            },
            'polling_details': {
                'polling_station_name': get_val('polling_details_polling_station_name', ''),
                'polling_station_number': get_val('polling_details_polling_station_number', ''),
                'polling_station_address': get_val('polling_details_polling_station_address', ''),
                'polling_station_type': get_val('polling_details_polling_station_type', ''),
                'auxiliary_polling_station_count': get_val('polling_details_auxiliary_polling_station_count', ''),
                'sections': get_val('polling_details_sections', ''),
            },
            'detailed_elector_summary': {
                'net_total': {
                    'male': get_val('detailed_elector_summary_net_total_male', ''),
                    'female': get_val('detailed_elector_summary_net_total_female', ''),
                    'third_gender': get_val('detailed_elector_summary_net_total_third_gender', ''),
                    'total': get_val('detailed_elector_summary_net_total_total', ''),
                },
                'serial_number_range': {
                    'start': get_val('detailed_elector_summary_serial_number_range_start', ''),
                    'end': get_val('detailed_elector_summary_serial_number_range_end', ''),
                }
            },
            'authority_verification': {
                'signature_present': get_val('authority_verification_signature_present', ''),
            },
            'output_identifier': get_val('output_identifier', ''),
            
            # NEW: Flat metadata fields (for new schema columns)
            'language_detected': get_val('language_detected', '[]'),  # May be JSON string or list
            'total_flat': int(get_val(['total', 'detailed_elector_summary_net_total_total'], 0) or 0) or None,  # Flat total field
        }
        
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
        
    def insert_metadata(self, metadata: Dict) -> Literal['created']:
        """
        Insert new metadata record in database.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            'created' when new record is inserted
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would insert metadata: {metadata['pdf_name']}")
            return 'created'
            
        query = """
            INSERT INTO metadata (
                document_id, pdf_name, state, year, revision_type,
                qualifying_date, publication_date, roll_type, roll_identification,
                total_pages, total_voters_extracted, 
                town_or_village, main_town_or_village, ward_number, post_office,
                police_station, taluk_or_block, subdivision, district, pin_code, panchayat_name,
                language_detected, total,
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
                %s, %s,
                %s, %s, %s,
                %s
            )
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
                Json(parse_json_field(metadata.get('language_detected', '[]'))),  # NEW: Flat field
                metadata.get('total_flat'),  # NEW: Flat field
                Json(parse_json_field(metadata['constituency_details'])),
                Json(parse_json_field(metadata['administrative_address'])),
                Json(parse_json_field(metadata['polling_details'])),
                Json(parse_json_field(metadata['detailed_elector_summary'])),
                Json(parse_json_field(metadata['authority_verification'])),
                metadata['output_identifier']
            ))
            
        self.db_conn.commit()
        
        logger.info(f"Inserted metadata: {metadata['pdf_name']} (document_id: {metadata['document_id']})")
        
        return 'created'
        
    def insert_voters(self, voters: List[Dict]) -> int:
        """
        Insert voter records into database.
        
        Args:
            voters: List of voter dictionaries
            
        Returns:
            Number of voters inserted
        """
        if not voters:
            return 0
            
        if self.dry_run:
            logger.info(f"[DRY RUN] Would insert {len(voters)} voters")
            return len(voters)
        
        # Get document_id from first voter (all voters should have same document_id)
        document_id = voters[0].get('document_id') if voters else None
        if not document_id:
            logger.error("Cannot insert voters: no document_id found")
            return 0
            
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
        logger.info(f"  Inserted {len(voters)} voters for document {document_id}")
        
        return len(voters)
        
        
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

    def sync_document(self, csv_pair: CSVPair, bucket: str) -> Optional[SyncOperation]:
        """
        Sync a single document (metadata + voters) from S3 to database.
        Only inserts new documents - skips existing ones.
        
        Args:
            csv_pair: CSV pair to sync
            bucket: S3 bucket name
            
        Returns:
            SyncOperation object with details of the operation, or None if failed
        """
        logger.info(f"Syncing document: {csv_pair.pdf_name} (assembly: {csv_pair.assembly_folder})")
        
        # Ensure connection before starting a transaction
        self._ensure_connection()
        
        try:
            # OPTIMIZATION: Check if document exists by pdf_name BEFORE downloading anything
            check_query = "SELECT document_id FROM metadata WHERE pdf_name = %s"
            with self.db_conn.cursor() as cur:
                cur.execute(check_query, (csv_pair.pdf_name,))
                exists = cur.fetchone() is not None
            
            if exists:
                # Document already exists, skip without downloading
                logger.info(f"  Document already exists in database (checked by pdf_name), skipping: {csv_pair.pdf_name}")
                
                # Use pdf_name as document_id for skipped entries (since we didn't download metadata)
                sync_op = SyncOperation(
                    document_id=csv_pair.pdf_name,  # Fallback to pdf_name
                    pdf_name=csv_pair.pdf_name,
                    operation='skipped',
                    assembly_folder=csv_pair.assembly_folder,
                    timestamp=datetime.now().isoformat(),
                    voters_count=0
                )
                
                return sync_op
            
            # Document is new, proceed with download and insert
            # Build cache keys for cached files
            cache_metadata_key = f"{csv_pair.assembly_folder}/{csv_pair.pdf_name}_metadata.csv"
            cache_voters_key = f"{csv_pair.assembly_folder}/{csv_pair.pdf_name}_voters.csv"
            
            # Download and parse metadata CSV (with caching)
            metadata_csv = self.download_csv_from_s3(bucket, csv_pair.metadata_key, cache_metadata_key)
            metadata = self.parse_metadata_csv(metadata_csv, csv_pair.pdf_name)
            
            # Download and parse voters CSV (with caching)
            voters_csv = self.download_csv_from_s3(bucket, csv_pair.voters_key, cache_voters_key)
            voters = self.parse_voters_csv(voters_csv, default_document_id=metadata['document_id'])
            
            # Insert into database (only in non-dry-run mode)
            operation = self.insert_metadata(metadata)
            voters_count = self.insert_voters(voters)
            
            logger.info(f"Successfully synced: {csv_pair.pdf_name} ({operation}, {voters_count} voters)")
            
            # Create operation record
            sync_op = SyncOperation(
                document_id=metadata['document_id'],
                pdf_name=csv_pair.pdf_name,
                operation=operation,
                assembly_folder=csv_pair.assembly_folder,
                timestamp=datetime.now().isoformat(),
                voters_count=voters_count
            )
            
            return sync_op
            
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
            return None
                
    def sync_all(self, bucket: str, prefix: str, limit: Optional[int] = None, s3_output: Optional[str] = None):
        """
        Sync new documents from S3 to database (skips existing documents).
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix (folder path)
            limit: Optional limit on number of documents to sync
            s3_output: Optional S3 path to upload the sync report JSON
        """
        # List CSV files in S3 (now discovers sub-folders)
        csv_pairs = self.list_s3_csv_files(bucket, prefix)
        
        logger.info(f"Found {len(csv_pairs)} total documents to process")
        
        if limit:
            csv_pairs = csv_pairs[:limit]
            logger.info(f"Limiting to {len(csv_pairs)} documents")
        
        # Track all operations
        operations: List[SyncOperation] = []
        
        # Sync each document
        for i, csv_pair in enumerate(csv_pairs, 1):
            logger.info(f"[{i}/{len(csv_pairs)}] Processing: {csv_pair.pdf_name} (assembly: {csv_pair.assembly_folder})")
            
            # Retry loop for single document
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    sync_op = self.sync_document(csv_pair, bucket)
                    if sync_op:
                        operations.append(sync_op)
                    break  # Success, move to next document
                except psycopg2.OperationalError:
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying connection for {csv_pair.pdf_name} (Attempt {attempt+2}/{max_retries})")
                        import time
                        time.sleep(5)  # Wait before retry
                        continue
                    else:
                        logger.error(f"Failed to sync {csv_pair.pdf_name} after {max_retries} connection attempts.")
                except Exception:
                    break  # Other errors handled inside sync_document, move to next
        
        logger.info(f"Sync complete! Processed {len(operations)} documents")
        
        # Generate and save sync report
        if operations:
            report_path = self.generate_sync_report(operations, s3_output, bucket)
            logger.info(f"Sync report saved to: {report_path}")
        else:
            logger.warning("No operations to report")
        
        # Clean up cache directory after normal run (not in dry-run)
        if not self.dry_run and self.cache_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.cache_dir)
                logger.info(f"Cleaned up cache directory: {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up cache directory: {e}")
    
    def generate_sync_report(self, operations: List[SyncOperation], s3_output: Optional[str], bucket: str) -> str:
        """
        Generate and save sync report as JSON file.
        
        Args:
            operations: List of SyncOperation objects
            s3_output: Optional S3 path to upload the report
            bucket: S3 bucket name
            
        Returns:
            Path to the saved report file
        """
        # Create sync_logs directory if it doesn't exist
        logs_dir = Path("sync_logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sync_report_{timestamp}.json"
        file_path = logs_dir / filename
        
        # Count operations
        created_count = sum(1 for op in operations if op.operation == 'created')
        updated_count = sum(1 for op in operations if op.operation == 'updated')
        skipped_count = sum(1 for op in operations if op.operation == 'skipped')
        
        # Get unique assemblies
        assemblies = sorted(list(set(op.assembly_folder for op in operations)))
        
        # Build report
        report = {
            "sync_timestamp": datetime.now().isoformat(),
            "total_documents_processed": len(operations),
            "documents_created": created_count,
            "documents_updated": updated_count,
            "documents_skipped": skipped_count,
            "assemblies_processed": assemblies,
            "operations": [
                {
                    "document_id": op.document_id,
                    "pdf_name": op.pdf_name,
                    "operation": op.operation,
                    "assembly_folder": op.assembly_folder,
                    "voters_count": op.voters_count,
                    "timestamp": op.timestamp
                }
                for op in operations
            ]
        }
        
        # Save to local file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Sync report saved locally: {file_path}")
        logger.info(f"  - Created: {created_count} documents")
        logger.info(f"  - Updated: {updated_count} documents")
        logger.info(f"  - Skipped: {skipped_count} documents")
        logger.info(f"  - Assemblies: {len(assemblies)}")
        
        # Upload to S3 if requested
        if s3_output:
            try:
                # Parse S3 output path (e.g., "s3://bucket/path/" or just "path/")
                s3_key = s3_output.replace('s3://', '').strip('/')
                # If bucket is included in path, extract it
                if '/' in s3_key:
                    parts = s3_key.split('/', 1)
                    if not s3_key.startswith(bucket):
                        # Use provided bucket and treat s3_output as key prefix
                        s3_key = f"{s3_key}/{filename}"
                    else:
                        # Extract bucket from path
                        bucket = parts[0]
                        s3_key = f"{parts[1]}/{filename}" if len(parts) > 1 else filename
                else:
                    s3_key = f"{s3_key}/{filename}"
                
                # Upload file
                with open(file_path, 'rb') as f:
                    self.s3_client.put_object(
                        Bucket=bucket,
                        Key=s3_key,
                        Body=f.read(),
                        ContentType='application/json'
                    )
                
                logger.info(f"Sync report uploaded to S3: s3://{bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload report to S3: {e}", exc_info=True)
        
        return str(file_path)


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
    parser.add_argument(
        '--s3-output',
        type=str,
        help="S3 path to upload sync report JSON (e.g., s3://bucket/path/ or just path/)"
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
                limit=args.limit,
                s3_output=args.s3_output
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
