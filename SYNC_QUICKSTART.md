# S3 to Database Sync - Quick Start Guide

## Overview

This guide helps you sync CSV files from S3 to your PostgreSQL database. The sync ensures that data extracted and stored as CSVs in S3 is also available in your database.

## Files Created

1. **sync_s3_to_db.py** - Main sync script
2. **verify_sync_setup.py** - Pre-flight verification script
3. **sync_helper.py** - Shows common usage patterns
4. **SYNC_S3_TO_DB.md** - Complete documentation

## Quick Start (3 Steps)

### Step 1: Verify Setup

Run the verification script to check your configuration:

```bash
python verify_sync_setup.py
```

This will:
- Test S3 connection
- List available CSV files
- Check database connection
- Show what would be synced
- Download and validate a sample CSV

**If this fails**, check your `.env` file for correct credentials.

### Step 2: Dry Run

Preview what would be synced without making changes:

```bash
python sync_s3_to_db.py --dry-run --limit 5
```

Review the output to ensure everything looks correct.

### Step 3: Sync Data

Start with a small batch:

```bash
python sync_s3_to_db.py --limit 10
```

If successful, run the full sync:

```bash
python sync_s3_to_db.py
```

## Default Configuration

The scripts use these defaults (configurable via command-line arguments):

- **S3 Bucket**: `264676382451-eci-download2026`
- **S3 Prefix**: `1/S22/extraction_results/`

## File Naming Convention

### CSV Files in S3
```
Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_metadata.csv
Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_voters.csv
```

### Database pdf_name
```
Tamil Nadu-(S22)_Coimbatore (North)-(AC118_1
```

**Note**: The script automatically handles the conversion between these formats.

## Required Environment Variables

Ensure these are set in your `.env` file:

```env
# S3 Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-south-1

# Database Configuration
DB_HOST=your-rds-endpoint.aws.com
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_username
DB_PASSWORD=your_password
```

## Common Commands

### Show usage patterns
```bash
python sync_helper.py
```

### Verify everything is configured
```bash
python verify_sync_setup.py
```

### Preview sync (no changes)
```bash
python sync_s3_to_db.py --dry-run
```

### Sync with limit
```bash
python sync_s3_to_db.py --limit 10
```

### Full sync
```bash
python sync_s3_to_db.py
```

### Custom S3 location
```bash
python sync_s3_to_db.py --bucket my-bucket --prefix my/path/
```

## What Gets Synced

For each document:

1. **Metadata** (1 row) - Document-level information:
   - PDF name, state, year
   - Administrative details
   - Constituency information
   - Total pages, total voters

2. **Voters** (many rows) - Individual voter records:
   - EPIC number, name, age, gender
   - House number, address
   - Relation information (father/mother/husband)
   - Page and sequence information

## How It Works

```
┌─────────────┐
│   S3 CSVs   │
│  (Source)   │
└──────┬──────┘
       │
       │ 1. List files
       ▼
┌─────────────┐
│Sync Script  │
│             │
└──────┬──────┘
       │
       │ 2. Check existing
       ▼
┌─────────────┐
│ PostgreSQL  │
│  metadata   │
└──────┬──────┘
       │
       │ 3. Find missing
       ▼
┌─────────────┐
│  Download   │
│  & Parse    │
└──────┬──────┘
       │
       │ 4. Insert data
       ▼
┌─────────────┐
│ PostgreSQL  │
│metadata +   │
│voters tables│
└─────────────┘
```

## Troubleshooting

### Connection Issues

**S3 connection failed**
- Check `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env`
- Verify bucket name and region

**Database connection failed**
- Check `DB_HOST`, `DB_USER`, `DB_PASSWORD` in `.env`
- Ensure database is accessible from your network

### No CSV Files Found

- Verify bucket name and prefix
- Check S3 permissions
- Ensure CSV files follow naming convention

### Parse Errors

- Check CSV format matches expected columns
- Ensure CSV has headers
- Verify encoding (should be UTF-8)

## Monitoring Progress

The sync script provides detailed logging:

```
INFO - Found 75 complete CSV pairs
INFO - Found 50 existing documents in database
INFO - Found 25 missing documents to sync
INFO - [1/25] Processing: Tamil Nadu-(S22)_Coimbatore (North)-(AC118_1
INFO - Inserted metadata: Tamil Nadu-(S22)_Coimbatore (North)-(AC118_1
INFO - Inserted 1234 voters
INFO - Successfully synced: Tamil Nadu-(S22)_Coimbatore (North)-(AC118_1
...
INFO - Sync complete! Processed 25 documents
```

## Safety Features

- **Dry run mode**: Preview without making changes
- **Idempotent**: Can be re-run safely (uses `ON CONFLICT DO NOTHING`)
- **Transaction-based**: Each document is a separate transaction
- **Error handling**: Continues with next document on error
- **Limit option**: Test with small batches first

## Performance

- Processes one document at a time (safe, resumable)
- Skips already-synced documents
- Uses batch inserts for voters
- Network-efficient (only downloads missing documents)

## Next Steps After Sync

Once sync is complete, you can query your data:

```sql
-- Count documents
SELECT COUNT(*) FROM metadata;

-- Count voters
SELECT COUNT(*) FROM voters;

-- View recent documents
SELECT pdf_name, state, district, total_voters_extracted 
FROM metadata 
ORDER BY created_at DESC 
LIMIT 10;

-- Get voters for a document
SELECT serial_no, epic_no, name, age, gender 
FROM voters 
WHERE document_id = 'your-document-id'
ORDER BY sequence_in_page;
```

## Support

For detailed information, see:
- **SYNC_S3_TO_DB.md** - Complete documentation
- **schema.sql** - Database schema reference
- **src/persistence/postgres.py** - Database implementation
- **src/utils/s3_utils.py** - S3 utilities

## Example Session

```bash
# 1. Verify setup
$ python verify_sync_setup.py
✓ All checks passed!

# 2. Preview what would be synced
$ python sync_s3_to_db.py --dry-run --limit 5
[DRY RUN] Would insert metadata: Tamil Nadu-(S22)_...
[DRY RUN] Would insert 1234 voters
...

# 3. Test with small batch
$ python sync_s3_to_db.py --limit 5
Inserted metadata: Tamil Nadu-(S22)_...
Inserted 1234 voters
Successfully synced: Tamil Nadu-(S22)_...
...

# 4. Run full sync
$ python sync_s3_to_db.py
Sync complete! Processed 25 documents
```

## Success Criteria

After running the sync, verify:

1. **No errors** in output
2. **Metadata count** matches expected number of documents
3. **Voters count** matches total_voters_extracted sum
4. **Sample queries** return expected data

---

**Ready to start?** Run `python verify_sync_setup.py` first!
