# Electoral Roll PDF Processing

A Python application for extracting voter information from Indian Electoral Roll PDFs using OCR and AI.

## Features

- **High-Accuracy OCR**: Uses advanced **Tamil OCR** library (`ocr_tamil`) and Tesseract for superior recognition of Tamil and English text.
- **Robust Field Extraction**: Multi-stage extraction with fallback mechanisms:
  - **EPIC Number**: Regex search -> Region of Interest (ROI) Crop -> **AI Fallback** (if OCR fails).
  - **Age/Gender**: logic extraction -> AI Fallback.
- **AI-Powered Recovery**: Automatically uses AI (Groq/Gemini) to recover missing data (EPIC, Age) or detect "Deleted" status.
- **Smart Validation**: 
  - Validates House Numbers (alphanumeric + standard separators only).
  - Auto-corrects Serial Numbers based on document sequence.
- **Performance Optimized**: 
  - **Batch Inference**: Processes multiple voters in parallel using merged image batches.
  - **Front-Page Metadata**: Restricts costly AI metadata extraction to the front page only.
- **PDF Extraction**: Convert PDF pages to high-resolution images.
- **AWS S3 Integration**: Process PDFs directly from S3 buckets.
- **Modular Architecture**: Easy to extend and maintain.

## Recent Updates (Jan 2026)
- **EPIC AI Fallback**: Implemented a 3-layer extraction strategy for EPIC numbers. If standard OCR fails, the system crops the specific EPIC region and, if needed, sends it to the AI for precise character recognition.
- **Enhanced OCR Integration**: Switched to optimized OCR engines for improved handling of bi-lingual (Tamil/English) voter cards.
- **Data Quality Logic**: 
  - Fixed House Number extraction to strictly allow `[A-Z0-9/-]` characters, removing noise.
  - Replaced OCR-based Serial Numbering with reliable document-level sequencing to prevent "1, 1, 1" errors.
- **Optimized Cropping**: Refined `id_field_cropper` to target specific fields more accurately.

## Project Structure

```
Electorials/
├── main.py                 # Unified entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment configuration template
├── prompt.md              # AI prompt for metadata extraction
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging with DEBUG support
│   ├── exceptions.py      # Custom exceptions
│   │
│   ├── models/            # Data models
│   │   ├── voter.py       # Voter data model
│   │   ├── metadata.py    # Document metadata model
│   │   ├── processing_stats.py  # Timing/cost tracking
│   │   └── document.py    # ProcessedDocument container
│   │
│   ├── processors/        # Processing components
│   │   ├── base.py        # Base processor class
│   │   ├── pdf_extractor.py     # PDF to images
│   │   ├── metadata_extractor.py # AI metadata
│   │   ├── image_cropper.py     # Voter box cropping
│   │   ├── id_field_cropper.py  # Specific field cropper
│   │   ├── image_merger.py      # Image batch merger
│   │   └── ocr_processor.py     # OCR extraction
│   │
│   ├── persistence/       # Data storage
│   │   ├── json_store.py  # JSON file storage
│   │   └── repository.py  # Repository pattern
│   │
│   └── utils/             # Utility functions
│       ├── file_utils.py  # File operations
│       ├── image_utils.py # Image processing
│       ├── s3_utils.py    # AWS S3 operations
│       └── timing.py      # Timing utilities
│
├── pdfs/                  # Input PDF files
├── extracted/             # Extracted data per PDF
│   └── <pdf_name>/
│       ├── manifest.json
│       ├── images/        # Page images
│       ├── crops/         # Cropped voter boxes
│       │   └── page-XXX/
│       └── output/        # Processing output
│           ├── <pdf_name>.json
│           ├── <pdf_name>-metadata.json
│           └── page_wise/
└── logs/                  # Log files
```

## Installation

### 1. Clone and Setup

```bash
cd Electorials
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

**Windows:**
Download from https://github.com/UB-Mannheim/tesseract/wiki

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-tam
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your AI API key and preferences
```

## Usage

### Process All PDFs

```bash
python main.py
```

### Process Specific PDFs

```bash
python main.py path/to/roll1.pdf path/to/roll2.pdf
```

### Process PDFs from S3

You can process PDFs directly from AWS S3 buckets:

```bash
# Single file from S3
python main.py s3://my-bucket/electoral-rolls/roll1.pdf

# Multiple files from S3
python main.py s3://bucket/roll1.pdf s3://bucket/roll2.pdf s3://bucket/roll3.pdf

# Mix local and S3 files
python main.py local.pdf s3://bucket/remote.pdf ./pdfs/another.pdf

# Using HTTPS S3 URLs
python main.py https://my-bucket.s3.ap-south-1.amazonaws.com/roll.pdf
```

**Supported S3 URL formats:**
- `s3://bucket-name/path/to/file.pdf`
- `https://bucket-name.s3.region.amazonaws.com/path/to/file.pdf`
- `https://s3.region.amazonaws.com/bucket-name/path/to/file.pdf`

### Run Specific Steps

You can run individual processing steps using the `--step` argument:

```bash
# Extract images only (Step 1)
python main.py --step extract

# Extract metadata using AI (Step 2)
python main.py --step metadata

# Crop voter boxes (Step 3)
python main.py --step crop

# Crop specific ID fields (Sequence, EPIC, HouseNo)
python main.py --step id-crop

# Merge ID field crops into batches
python main.py --step id-merge

# Merge voter images into batches for faster OCR
python main.py --step merge

# Extract ID data (Sequence, EPIC, HouseNo) using AI
python main.py --step id-extract

# Run OCR only (Step 5)
python main.py --step ocr

# Export to CSV (Step 6)
python main.py --step csv

# Run everything (Default)
python main.py --step all
```

**Available steps:** `extract`, `metadata`, `crop`, `field-crop`, `id-crop`, `header`, `merge`, `top-merge`, `id-merge`, `id-extract`, `ocr`, `csv`, `all`.

### Process Existing Extracted Folders

```bash
# List extracted folders
python main.py --list

# Process specific folder
python main.py --step ocr --folder "2025-EROLLGEN-S22-114-FinalRoll-Revision1-TAM-1-WI"
```

### Debug Mode

```bash
# Enable debug logging
DEBUG=1 python main.py

# Or set in .env:
# DEBUG=1
```

### Additional Options

```bash
python main.py --help

Options:
  --step {extract,metadata,crop,field-crop,id-crop,header,merge,top-merge,id-merge,id-extract,ocr,csv,all}
                                             Run specific step (default: all)
  --folder FOLDER                            Process specific folder
  --list                                     List extracted folders
  --force                                    Force reprocessing
  --limit N                                  Limit to first N items
  --dpi DPI                                  PDF rendering DPI (default: 200)
  --languages LANG                           OCR languages (default: eng+tam)
  --diagram-filter {auto,on,off}             Diagram filter mode
  --skip-metadata                            Skip AI metadata extraction
  --dump-raw-ocr                             Dump raw OCR for debugging
  --use-crops                                Use individual crop images instead of merged batches (slower)
  --use-tamil-ocr                            Use Tamil OCR (ocr_tamil) instead of Tesseract
  --use-ai-ocr                               Use AI (Groq/Vision) for OCR extraction
  --no-csv                                   Disable CSV export
```

## Output Format

### Combined Output (`<pdf_name>.json`)

```json
{
  "document_id": "2025-EROLLGEN-S22-114-...",
  "pdf_name": "2025-EROLLGEN-S22-114-...",
  "status": "completed",
  "pages_count": 50,
  "total_voters": 1234,
  "valid_voters": 1200,
  "metadata": {
    "state": "Tamil Nadu",
    "ac_no": "114",
    "ac_name": "Mylapore",
    "part_no": "1",
    "revision_year": "2025"
  },
  "timing": {
    "total_time_seconds": 450.5,
    "extraction_seconds": 30.2,
    "cropping_seconds": 60.1,
    "ocr_seconds": 360.2
  },
  "records": [
    {
      "serial_no": "1",
      "epic_no": "ABC1234567",
      "name": "John Doe",
      "relation_type": "father",
      "relation_name": "Richard Doe",
      "house_no": "123",
      "age": 45,
      "gender": "Male",
      "page_id": "page-003",
      "sequence_in_page": 1,
      "sequence_in_document": 1
    }
  ]
}
```

### Metadata Output (`<pdf_name>-metadata.json`)

```json
{
  "state": "Tamil Nadu",
  "district": "Chennai",
  "ac_no": "114",
  "ac_name": "Mylapore",
  "pc_no": "22",
  "pc_name": "Chennai South",
  "part_no": "1",
  "language": "Tamil",
  "revision_year": "2025",
  "revision_type": "Final Roll",
  "total_male_electors": 500,
  "total_female_electors": 550,
  "total_third_gender_electors": 2,
  "total_electors": 1052,
  "ai_metadata": {
    "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "usage": {"prompt_tokens": 1500, "completion_tokens": 200}
  }
}
```

## Environment Variables

### General

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug logging | `0` |
| `PDFS_DIR` | Input PDF directory | `./pdfs` |
| `EXTRACTED_DIR` | Output directory | `./extracted` |
| `LOGS_DIR` | Log directory | `./logs` |

### AI Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_PROVIDER` | AI provider (groq, gemini, openai) | `Groq` |
| `AI_API_KEY` | API key for metadata extraction | Required |
| `AI_MODEL` | AI model name | `meta-llama/llama-4-maverick-17b-128e-instruct` |
| `AI_BASE_URL` | API base URL | Provider default |
| `AI_TIMEOUT_SEC` | Request timeout | `120` |

### AWS S3 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | - |
| `AWS_SESSION_TOKEN` | Session token (temporary creds) | - |
| `AWS_REGION` | AWS region | `ap-south-1` |
| `S3_DEFAULT_BUCKET` | Default S3 bucket | - |
| `S3_DOWNLOAD_DIR` | Local download directory | System temp |
| `S3_CONNECT_TIMEOUT` | Connection timeout (seconds) | `10` |
| `S3_READ_TIMEOUT` | Read timeout (seconds) | `60` |
| `S3_MAX_RETRIES` | Max retry attempts | `3` |

> **Note:** S3 credentials can also be provided via IAM roles, `AWS_PROFILE`, or the default AWS credential chain.

## Development

### Running Individual Modules

The legacy scripts are still available for backward compatibility:

```bash
# Extract PDFs
python extract_source.py

# Extract metadata
python meta_info_capture_ai.py

# Crop voter boxes
python crop_voters_for_ocr.py

# Run OCR
python ocr_processor.py
```

### Project Architecture

```
                    ┌─────────────┐
                    │   main.py   │
                    │ (CLI Entry) │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Config  │ │  Logger  │ │  Models  │
        └──────────┘ └──────────┘ └──────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │    PDF    │    │  Metadata │    │   Image   │
   │ Extractor │───▶│ Extractor │───▶│  Cropper  │
   └───────────┘    └───────────┘    └───────────┘
                                           │
                                           ▼
                                    ┌───────────┐
                                    │    OCR    │
                                    │ Processor │
                                    └───────────┘
                                           │
                                           ▼
                                    ┌───────────┐
                                    │   JSON    │
                                    │   Store   │
                                    └───────────┘
```

## License

MIT License