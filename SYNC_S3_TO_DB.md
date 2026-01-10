# S3 to Database Sync Script

## Overview

The `sync_s3_to_db.py` script synchronizes CSV data from your S3 bucket to your PostgreSQL database. It's designed to ensure that data extracted and stored as CSVs in S3 is also available in your database for querying and analysis.

## How It Works

1. **Lists CSV files** in the specified S3 bucket and prefix
2. **Pairs metadata and voters CSV files** based on naming convention
3. **Checks metadata table** in PostgreSQL to identify missing documents
4. **Downloads missing CSVs** from S3
5. **Inserts data** into metadata and voters tables

## File Naming Convention

The script expects CSV files to follow this naming pattern:

### In S3:
```
Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_metadata.csv
Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_voters.csv
```

### In Database (pdf_name):
```
Tamil Nadu-(S22)_Coimbatore (North)-(AC118_1
```

**Note:** The script automatically converts between these formats. The CSV format has `(AC118)_1` while the database format has `(AC118_1` (missing closing parenthesis).

## Usage

### Basic Usage

Sync all missing documents:

```bash
python sync_s3_to_db.py
```

### Dry Run

Preview what would be synced without making any changes:

```bash
python sync_s3_to_db.py --dry-run
```

### Limit Sync

Sync only the first N documents (useful for testing):

```bash
python sync_s3_to_db.py --limit 10
```

### Custom S3 Location

Specify a different bucket or prefix:

```bash
python sync_s3_to_db.py --bucket my-bucket --prefix path/to/csvs/
```

### All Options Combined

```bash
python sync_s3_to_db.py --dry-run --limit 5 --bucket 264676382451-eci-download2026 --prefix 1/S22/extraction_results/
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dry-run` | Show what would be synced without making database changes | False |
| `--limit N` | Sync only first N documents | None (all) |
| `--bucket` | S3 bucket name | `264676382451-eci-download2026` |
| `--prefix` | S3 folder path | `1/S22/extraction_results/` |

## Prerequisites

### 1. Environment Variables

Ensure your `.env` file has the required configuration:

```env
# S3 Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-south-1
S3_DEFAULT_BUCKET=264676382451-eci-download2026

# Database Configuration
DB_HOST=your-rds-endpoint.aws.com
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_username
DB_PASSWORD=your_password
DB_SCHEMA=public
```

### 2. Python Dependencies

The script uses existing dependencies from the project. Ensure you have installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `boto3` - For S3 access
- `psycopg2` or `psycopg2-binary` - For PostgreSQL
- `python-dotenv` - For environment configuration

### 3. Database Tables

The script requires the following tables to exist (they should already be created if you've run the main processing):

- `metadata` - Stores document-level information
- `voters` - Stores individual voter records

If tables don't exist, create them using:

```bash
psql -h your-host -U your-user -d your-db -f schema.sql
```

## CSV Format

### Metadata CSV Columns

Expected columns in `*_metadata.csv`:
- `document_id`
- `pdf_name`
- `state`
- `year`
- `revision_type`
- `qualifying_date`
- `publication_date`
- `roll_type`
- `roll_identification`
- `total_pages`
- `total_voters_extracted`
- `town_or_village`
- `main_town_or_village`
- `ward_number`
- `post_office`
- `police_station`
- `taluk_or_block`
- `subdivision`
- `district`
- `pin_code`
- `panchayat_name`
- `constituency_details` (JSON string)
- `administrative_address` (JSON string)
- `polling_details` (JSON string)
- `detailed_elector_summary` (JSON string)
- `authority_verification` (JSON string)
- `output_identifier`

### Voters CSV Columns

Expected columns in `*_voters.csv`:
- `id`
- `document_id`
- `serial_no`
- `epic_no`
- `name`
- `relation_type`
- `relation_name`
- `father_name`
- `mother_name`
- `husband_name`
- `other_name`
- `house_no`
- `age`
- `gender`
- `street_names_and_numbers`
- `part_no`
- `assembly`
- `page_id`
- `sequence_in_page`
- `epic_valid`
- `deleted`

## How to Run

### Step 1: Test with Dry Run

First, do a dry run to see what would be synced:

```bash
python sync_s3_to_db.py --dry-run --limit 5
```

Review the output to ensure:
- S3 connection is successful
- CSV files are found and paired correctly
- Missing documents are identified correctly

### Step 2: Sync Small Batch

Sync a small number of documents to verify everything works:

```bash
python sync_s3_to_db.py --limit 10
```

Check the database to ensure records were inserted correctly:

```sql
SELECT pdf_name, total_voters_extracted FROM metadata ORDER BY created_at DESC LIMIT 10;
```

### Step 3: Full Sync

Once verified, run the full sync:

```bash
python sync_s3_to_db.py
```

## Logging

The script uses the project's logger configuration and will output:
- INFO: Progress updates, successful operations
- WARNING: Missing voter CSVs, conversion issues
- ERROR: Failed downloads, database errors

Logs are displayed on console and saved to the `logs/` directory.

## Error Handling

The script includes error handling for:
- **S3 connection failures**: Checks credentials and bucket access
- **Database connection failures**: Validates connection parameters
- **Missing CSV pairs**: Warns if metadata CSV has no matching voters CSV
- **Parsing errors**: Logs detailed errors for CSV format issues
- **Insertion failures**: Rolls back transaction on error

If an error occurs for a specific document, the script logs it and continues with the next document.

## Troubleshooting

### Issue: "Failed to connect to S3"

**Solution:** Check your AWS credentials in `.env`:
```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### Issue: "Failed to connect to database"

**Solution:** Verify database credentials and host accessibility:
```env
DB_HOST=your-rds-endpoint.aws.com
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_username
DB_PASSWORD=your_password
```

### Issue: "No voters CSV found for document"

**Solution:** Ensure both metadata and voters CSV files exist in S3 with matching names.

### Issue: "CSV is empty" or "Missing columns"

**Solution:** Verify the CSV files in S3 have the expected format and headers.

### Issue: "Name conversion not working"

**Solution:** Check if your CSV filenames match the expected pattern. The script converts `(AC118)_1` to `(AC118_1`. If your naming is different, update the `_convert_to_db_format()` method.

## Advanced Usage

### Custom Conversion Logic

If your naming convention differs, modify the `_convert_to_db_format()` method in the script:

```python
def _convert_to_db_format(self, csv_name: str) -> str:
    # Add your custom conversion logic here
    return csv_name
```

### Processing Specific Documents

To sync specific documents, you can modify the script to accept a list of pdf_names:

```python
# Add after line 386 in sync_all()
if specific_docs:
    missing_pairs = [p for p in missing_pairs if p.pdf_name in specific_docs]
```

## Performance

- **Batch processing**: The script processes documents one at a time to manage memory
- **Transaction safety**: Each document is a separate transaction (metadata + voters)
- **Resumable**: Can be re-run; uses `ON CONFLICT DO NOTHING` for metadata
- **Network efficiency**: Downloads only missing documents

## Monitoring Progress

The script provides detailed progress information:

```
INFO - Listing CSV files in s3://bucket/prefix
INFO - Found 150 objects in S3
INFO - Found 75 complete CSV pairs
INFO - Fetching existing documents from database...
INFO - Found 50 existing documents in database
INFO - Found 25 missing documents to sync
INFO - [1/25] Processing: Tamil Nadu-(S22)_Coimbatore (North)-(AC118_1
INFO - Inserted metadata: Tamil Nadu-(S22)_Coimbatore (North)-(AC118_1
INFO - Inserted 1234 voters
INFO - Successfully synced: Tamil Nadu-(S22)_Coimbatore (North)-(AC118_1
...
INFO - Sync complete! Processed 25 documents
```

## Related Files

- `schema.sql` - Database schema definition
- `src/persistence/postgres.py` - PostgreSQL repository implementation
- `src/utils/s3_utils.py` - S3 utility functions
- `src/config.py` - Configuration loading

## Future Enhancements

Potential improvements:
- [ ] Parallel processing with multiprocessing
- [ ] Delta sync (only update changed records)
- [ ] Validation of CSV data before insertion
- [ ] More detailed reporting (summary statistics)
- [ ] Support for incremental syncs based on timestamp
