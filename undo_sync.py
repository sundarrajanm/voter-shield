"""
Undo a sync operation by deleting records from the database.

This script:
1. Reads a sync report JSON file from sync_logs/
2. Identifies all 'created' operations
3. Deletes voters records for those document_ids
4. Deletes metadata records for those document_ids
5. Generates a summary report

Usage:
    python undo_sync.py --report sync_logs/sync_report_20260114_130136.json [--dry-run]
    
Arguments:
    --report: Path to the sync report JSON file
    --dry-run: Show what would be deleted without making changes
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import psycopg2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


class SyncUndoer:
    """Undoes a sync operation by deleting created records."""
    
    def __init__(self, config: Config, dry_run: bool = False):
        """
        Initialize undoer.
        
        Args:
            config: Application configuration
            dry_run: If True, don't make any changes to the database
        """
        self.config = config
        self.dry_run = dry_run
        self.db_conn = None
        
    def __enter__(self):
        """Context manager entry."""
        self._connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()
        
    def _connect(self):
        """Establish database connection."""
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
            
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            raise Exception(f"Failed to connect to database: {e}")
            
    def _cleanup(self):
        """Clean up connections."""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Closed database connection")
            
    def load_sync_report(self, report_path: str) -> Dict:
        """
        Load sync report from JSON file.
        
        Args:
            report_path: Path to sync report JSON file
            
        Returns:
            Sync report dictionary
        """
        logger.info(f"Loading sync report: {report_path}")
        
        report_file = Path(report_path)
        if not report_file.exists():
            raise FileNotFoundError(f"Report file not found: {report_path}")
        
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        logger.info(f"Loaded report from {report['sync_timestamp']}")
        logger.info(f"  Total documents: {report['total_documents_processed']}")
        logger.info(f"  Created: {report['documents_created']}")
        logger.info(f"  Skipped: {report['documents_skipped']}")
        
        return report
        
    def undo_sync(self, report: Dict):
        """
        Undo sync operation by deleting created records.
        
        Args:
            report: Sync report dictionary
        """
        # Filter only 'created' operations
        created_ops = [
            op for op in report['operations']
            if op['operation'] == 'created'
        ]
        
        if not created_ops:
            logger.warning("No 'created' operations found in report. Nothing to undo.")
            return
        
        logger.info(f"Found {len(created_ops)} documents to delete")
        
        # Extract document_ids
        document_ids = [op['document_id'] for op in created_ops]
        
        # Count voters to be deleted
        voters_query = "SELECT COUNT(*) FROM voters WHERE document_id = ANY(%s)"
        with self.db_conn.cursor() as cur:
            cur.execute(voters_query, (document_ids,))
            voters_count = cur.fetchone()[0]
        
        logger.info(f"Will delete:")
        logger.info(f"  - {len(document_ids)} metadata records")
        logger.info(f"  - {voters_count} voter records")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would delete the above records (no changes made)")
            logger.info("\nSample document_ids that would be deleted:")
            for doc_id in document_ids[:5]:
                logger.info(f"  - {doc_id}")
            if len(document_ids) > 5:
                logger.info(f"  ... and {len(document_ids) - 5} more")
            return
        
        # Delete voters first (foreign key constraint)
        logger.info("Deleting voter records...")
        delete_voters_query = "DELETE FROM voters WHERE document_id = ANY(%s)"
        with self.db_conn.cursor() as cur:
            cur.execute(delete_voters_query, (document_ids,))
            deleted_voters = cur.rowcount
            logger.info(f"  Deleted {deleted_voters} voter records")
        
        # Delete metadata
        logger.info("Deleting metadata records...")
        delete_metadata_query = "DELETE FROM metadata WHERE document_id = ANY(%s)"
        with self.db_conn.cursor() as cur:
            cur.execute(delete_metadata_query, (document_ids,))
            deleted_metadata = cur.rowcount
            logger.info(f"  Deleted {deleted_metadata} metadata records")
        
        # Commit transaction
        self.db_conn.commit()
        logger.info("Undo complete! All changes committed.")
        
        # Summary
        logger.info("\n=== UNDO SUMMARY ===")
        logger.info(f"Metadata records deleted: {deleted_metadata}")
        logger.info(f"Voter records deleted: {deleted_voters}")
        logger.info(f"Total operations undone: {len(created_ops)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Undo a sync operation by deleting created records"
    )
    parser.add_argument(
        '--report',
        required=True,
        help="Path to sync report JSON file (e.g., sync_logs/sync_report_20260114_130136.json)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be deleted without making changes"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Run undo
    try:
        with SyncUndoer(config, dry_run=args.dry_run) as undoer:
            report = undoer.load_sync_report(args.report)
            undoer.undo_sync(report)
            
        logger.info("Undo operation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Undo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Undo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
