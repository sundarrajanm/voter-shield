"""
Test script to verify database connection validation logic.
This can be run to test if the database connection is working.

Usage:
    python check_db_connection.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.logger import get_logger, setup_logger

logger = get_logger(__name__)


def main():
    """Test database connection."""
    setup_logger()
    
    print("=" * 60)
    print("DATABASE CONNECTION TEST")
    print("=" * 60)
    print()
    
    # Load configuration
    try:
        config = Config()
        print(f"✓ Configuration loaded")
        print()
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return 1
    
    # Check if database is configured
    if not config.db.is_configured:
        print("⚠ Database is NOT configured in .env file")
        print()
        print("To configure database, set these environment variables:")
        print("  DB_HOST=localhost")
        print("  DB_PORT=5432")
        print("  DB_NAME=electoral_rolls")
        print("  DB_USER=postgres")
        print("  DB_PASSWORD=your_password")
        print("  DB_SSL_MODE=prefer")
        print("  DB_REQUIRED=true  # Set to true if database is mandatory")
        print()
        
        if config.db.required:
            print("✗ Database is REQUIRED (DB_REQUIRED=true) but not configured")
            return 1
        else:
            print("✓ Database is optional (DB_REQUIRED=false), application can proceed")
            return 0
    
    print(f"✓ Database is configured:")
    print(f"    Host: {config.db.host}")
    print(f"    Port: {config.db.port}")
    print(f"    Database: {config.db.name}")
    print(f"    User: {config.db.user}")
    print(f"    SSL Mode: {config.db.ssl_mode}")
    print(f"    Required: {config.db.required}")
    print()
    
    # Test connection
    print("Testing database connection...")
    print()
    
    try:
        from src.persistence.postgres import PostgresRepository
        
        db_repo = PostgresRepository(config.db)
        connection_successful = db_repo.test_connection()
        
        if connection_successful:
            print("=" * 60)
            print("✓ DATABASE CONNECTION SUCCESSFUL!")
            print("=" * 60)
            print()
            print("The database is reachable and ready for use.")
            print()
            return 0
        else:
            # test_connection returned False (DB not required)
            print("=" * 60)
            print("⚠ DATABASE CONNECTION FAILED!")
            print("=" * 60)
            print()
            print("However, DB_REQUIRED=false, so the application can proceed without database.")
            print()
            return 0
        
    except Exception as e:
        print("=" * 60)
        print("✗ DATABASE CONNECTION FAILED!")
        print("=" * 60)
        print()
        print(f"Error: {e}")
        print()
        
        if config.db.required:
            print("Database is REQUIRED (DB_REQUIRED=true)")
            print()
            print("Possible causes:")
            print("  - Database server is not running")
            print("  - Incorrect host/port in configuration")
            print("  - Incorrect username/password")
            print("  - Database does not exist")
            print("  - Network/firewall issues")
            print()
            return 1
        else:
            print("Database is optional (DB_REQUIRED=false)")
            print("The application can proceed without database.")
            print()
            return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
