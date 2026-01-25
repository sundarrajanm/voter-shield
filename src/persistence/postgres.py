"""
PostgreSQL repository implementation.
"""
from __future__ import annotations

import json
import logging
from typing import Optional, List, Any, Dict
import psycopg2
from psycopg2.extras import Json, execute_values
import uuid
from dataclasses import asdict

from ..config import DBConfig
from ..models import ProcessedDocument, DocumentMetadata, Voter

logger = logging.getLogger(__name__)

class PostgresRepository:
    """
    PostgreSQL repository for storing processed electoral roll data.
    
    Handles:
    - Connection management
    - Schema initialization
    - Data insertion (metadata and voters)
    """
    
    def __init__(self, config: DBConfig):
        """
        Initialize repository.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self._conn = None
        
    def _get_connection(self):
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    dbname=self.config.name,
                    user=self.config.user,
                    password=self.config.password,
                    sslmode=self.config.ssl_mode
                )
                self._conn.autocommit = False
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                raise
        return self._conn
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection is successful, False otherwise
            
        Raises:
            Exception if connection fails AND database is required
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result and result[0] == 1:
                    logger.info("Database connection test successful")
                    return True
                else:
                    logger.error("Database connection test failed: unexpected result")
                    if self.config.required:
                        raise Exception("Database connection test failed")
                    return False
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            if self.config.required:
                raise
            else:
                logger.warning("Database connection failed but DB_REQUIRED=false, continuing without database")
                return False
        
    def init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Create metadata table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        document_id TEXT PRIMARY KEY,
                        pdf_name TEXT NOT NULL,
                        state TEXT,
                        year INTEGER,
                        revision_type TEXT,
                        qualifying_date TEXT,
                        publication_date TEXT,
                        roll_type TEXT,
                        roll_identification TEXT,
                        total_pages INTEGER,
                        total_voters_extracted INTEGER,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Main administrative fields
                        town_or_village TEXT,
                        main_town_or_village TEXT,
                        ward_number TEXT,
                        post_office TEXT,
                        police_station TEXT,
                        taluk_or_block TEXT,
                        subdivision TEXT,
                        district TEXT,
                        pin_code TEXT,
                        panchayat_name TEXT,
                        
                        -- NEW: Flat metadata fields (migrated from nested structure)
                        language_detected JSONB DEFAULT '[]',
                        total INTEGER,
                        
                        -- DEPRECATED: Nested structures (kept for backward compatibility)
                        constituency_details JSONB DEFAULT '{}',
                        administrative_address JSONB DEFAULT '{}',
                        polling_details JSONB DEFAULT '{}',
                        detailed_elector_summary JSONB DEFAULT '{}',
                        authority_verification JSONB DEFAULT '{}',
                        output_identifier TEXT
                    );
                """)
                
                # Migration: Add output_identifier if not exists
                try:
                    cur.execute("ALTER TABLE metadata ADD COLUMN IF NOT EXISTS output_identifier TEXT;")
                except Exception:
                    conn.rollback()
                else:
                    conn.commit()
                
                # Create voters table with expanded fields to match CSV
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS voters (
                        id TEXT PRIMARY KEY,
                        document_id TEXT REFERENCES metadata(document_id),
                        serial_no TEXT,
                        epic_no TEXT,
                        name TEXT,
                        
                        -- Relation fields
                        relation_type TEXT,
                        relation_name TEXT,
                        father_name TEXT,
                        mother_name TEXT,
                        husband_name TEXT,
                        other_name TEXT,
                        
                        -- Address/Details
                        house_no TEXT,
                        age TEXT,
                        gender TEXT,
                        street_names_and_numbers TEXT,
                        part_no TEXT,
                        assembly TEXT,
                        
                        -- Metadata fields
                        page_id TEXT,
                        sequence_in_page INTEGER,
                        epic_valid BOOLEAN,
                        deleted TEXT,  -- Empty string = not deleted, 'true' = deleted
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_voters_document_id ON voters(document_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_voters_epic_no ON voters(epic_no);")
                
            conn.commit()
            logger.info("Database schema initialized")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize database: {e}")
            raise

    def save_document(self, document: ProcessedDocument) -> bool:
        """
        Save document metadata and voters to database.
        
        Args:
            document: Processed document with metadata and voters
            
        Returns:
            True if successful
        """
        if not document.metadata:
            logger.warning(f"No metadata found for document {document.id}, skipping DB save")
            return False
            
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # 1. Insert/Update Metadata
                self._save_metadata(cur, document)
                
                # 2. Insert Voters
                # Use document.all_voters property
                # But we also need page info for street/part/assembly which might not be in Voter object directly
                # ProcessedDocument.to_combined_json logic handles this mapping.
                # Let's reproduce that logic.
                
                # Iterate pages to get context
                voters_with_context = []
                for page in document.pages:
                    for voter in page.voters:
                         voters_with_context.append((voter, page))
                
                if voters_with_context:
                    self._save_voters(cur, document, voters_with_context)
                    
            conn.commit()
            logger.info(f"Successfully saved document {document.id} to database")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save document to database: {e}", exc_info=True)
            return False

    def _save_metadata(self, cur, document: ProcessedDocument):
        """Insert or update metadata record."""
        meta = document.metadata
        
        # Set total_voters_extracted from the actual extracted voters count
        meta.total_voters_extracted = document.total_voters
        
        # Prepare JSON fields
        pdf_name = document.pdf_name
        # If document_id in metadata is empty, fallback to document.id
        doc_id = meta.document_id if meta.document_id else document.id

        # Check for existing document with same pdf_name
        cur.execute("SELECT document_id FROM metadata WHERE pdf_name = %s", (pdf_name,))
        result = cur.fetchone()
        if result:
            existing_doc_id = result[0]
            if existing_doc_id != doc_id:
                logger.info(f"Found existing document with pdf_name '{pdf_name}' (id={existing_doc_id}). Updating this instead of creating {doc_id}.")
                doc_id = existing_doc_id
                # Update the document object too so voters get the right ID
                if meta:
                    meta.document_id = doc_id
        
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
                output_identifier,
                updated_at
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, 
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s,
                %s, %s, %s,
                %s,
                CURRENT_TIMESTAMP
            )
            ON CONFLICT (document_id) DO UPDATE SET
                pdf_name = EXCLUDED.pdf_name,
                state = EXCLUDED.state,
                year = EXCLUDED.year,
                revision_type = EXCLUDED.revision_type,
                qualifying_date = EXCLUDED.qualifying_date,
                publication_date = EXCLUDED.publication_date,
                roll_type = EXCLUDED.roll_type,
                roll_identification =EXCLUDED.roll_identification,
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
                language_detected = EXCLUDED.language_detected,
                total = EXCLUDED.total,
                constituency_details = EXCLUDED.constituency_details,
                administrative_address = EXCLUDED.administrative_address,
                polling_details = EXCLUDED.polling_details,
                detailed_elector_summary = EXCLUDED.detailed_elector_summary,
                authority_verification = EXCLUDED.authority_verification,
                output_identifier = EXCLUDED.output_identifier,
                updated_at = CURRENT_TIMESTAMP;
        """
        
        # Prepare Authority Verification (Exclude designation)
        auth_veri = asdict(meta.authority_verification) if hasattr(meta.authority_verification, 'designation') else meta.authority_verification
        if isinstance(auth_veri, dict):
            # Remove designation if present
            auth_veri.pop('designation', None)
        
        cur.execute(query, (
            doc_id,
            pdf_name,
            meta.state,
            meta.electoral_roll_year,
            meta.revision_type,
            meta.qualifying_date,
            meta.publication_date,
            meta.roll_type,
            meta.roll_identification,
            meta.total_pages,
            meta.total_voters_extracted,
            meta.administrative_address.town_or_village,
            meta.administrative_address.main_town_or_village,
            meta.administrative_address.ward_number,
            meta.administrative_address.post_office,
            meta.administrative_address.police_station,
            meta.administrative_address.taluk_or_block,
            meta.administrative_address.subdivision,
            meta.administrative_address.district,
            meta.administrative_address.pin_code,
            meta.administrative_address.panchayat_name,
            Json(meta.language_detected),  # NEW: Flat field
            meta.detailed_elector_summary.net_total.total,  # NEW: Flat field
            Json(meta.constituency_details.to_dict()),
            Json(meta.administrative_address.to_dict()),
            Json(meta.polling_details.to_dict()),
            Json(meta.detailed_elector_summary.to_dict()),
            Json(auth_veri),
            meta.output_identifier
        ))

    def _save_voters(self, cur, document: ProcessedDocument, voters_context: List[tuple]):
        """Insert voter records."""
        # Use document.id as document_id (should match what was used in metadata)
        doc_id = document.metadata.document_id if document.metadata and document.metadata.document_id else document.id
        
        # Delete existing voters for this document
        cur.execute("DELETE FROM voters WHERE document_id = %s", (doc_id,))
        
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
        for voter, page in voters_context:
            # Generate a UUID if ID is missing
            v_id = str(uuid.uuid4())
            
            # Fields from Voter dataclass
            serial = voter.serial_no
            epic = voter.epic_no
            name = voter.name
            rel_type = voter.relation_type
            rel_name = voter.relation_name
            house = voter.house_no
            age = voter.age
            gender = voter.gender
            is_valid = voter.epic_valid
            
            # Derived fields (matching CSV logic)
            r_type_lower = rel_type.lower()
            father = rel_name if "father" in r_type_lower else ""
            mother = rel_name if "mother" in r_type_lower else ""
            husband = rel_name if "husband" in r_type_lower else ""
            other = "" # Logic for other? CSV sets it empty.
            
            # Context fields
            street = page.street_name_and_number
            part_no = str(page.part_number) if page.part_number else ""
            assembly = page.assembly_constituency_number_and_name
            
            page_id = voter.page_id or page.page_id
            seq = voter.sequence_in_page
            deleted = voter.deleted  # Empty string = not deleted, 'true' = deleted
            
            values.append((
                v_id,
                doc_id,
                serial,
                epic,
                name,
                rel_type,
                rel_name,
                father,
                mother,
                husband,
                other,
                house,
                age,
                gender,
                street,
                part_no,
                assembly,
                page_id,
                seq,
                is_valid,
                deleted
            ))
            
        execute_values(cur, query, values)
