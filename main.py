#!/usr/bin/env python3
"""
Electoral Roll PDF Processing - Main Entry Point

Unified entry point for processing Electoral Roll PDFs:
- Extract images from PDFs
- Extract metadata using AI
- Crop voter information boxes
- OCR voter data extraction
- Generate combined output

Usage:
    python main.py                           # Process all PDFs
    python main.py path1.pdf path2.pdf       # Process specific PDFs
    python main.py s3://bucket/path.pdf      # Process PDF from S3
    python main.py --step extract            # Run specific step only
    python main.py --list                    # List available extracted folders

Environment Variables:
    DEBUG=1                 Enable debug logging
    AI_API_KEY             API key for AI metadata extraction
    AI_PROVIDER            AI provider (gemini, openai)
    AI_MODEL               AI model name
    
    # S3 Configuration (for S3 URLs)
    AWS_ACCESS_KEY_ID      AWS access key
    AWS_SECRET_ACCESS_KEY  AWS secret key
    AWS_REGION             AWS region (default: ap-south-1)
    S3_DEFAULT_BUCKET      Default S3 bucket

Examples:
    # Process all PDFs in pdfs/ directory
    python main.py

    # Process specific PDFs
    python main.py pdfs/roll1.pdf pdfs/roll2.pdf

    # Process PDF from S3
    python main.py s3://my-bucket/electoral-rolls/roll1.pdf

    # Run only extraction step
    python main.py --step extract

    # Run this to crop and OCR a specific extracted folder
    python main.py --step crop

    # Run only OCR on already extracted folders
    python main.py --step ocr

    # Enable debug mode
    DEBUG=1 python main.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Union
import shutil

from src.config import Config, get_config
from src.logger import get_logger, setup_logger
from src.models import ProcessedDocument, ProcessingStats
from src.persistence import JSONStore
from src.processors import (
    ProcessingContext,
    PDFExtractor,
    MetadataExtractor,
    ImageCropper,
    ImageMerger,
    OCRProcessor,
    HeaderExtractor,
    HeaderExtractor,
    CropTopMerger,
    IdFieldCropper,
    IdFieldMerger,
)
from src.utils.file_utils import iter_pdfs, iter_extracted_folders
from src.utils.timing import Timer
from src.utils.s3_utils import is_s3_url, download_from_s3, upload_to_s3


logger = get_logger("main")


def retry_database_operation(operation_func, operation_name: str, max_retries: int = 3, wait_seconds: int = 5) -> bool:
    """
    Retry a database operation with exponential backoff.
    
    Args:
        operation_func: Function to execute (should return True on success, False on failure)
        operation_name: Name of the operation for logging
        max_retries: Maximum number of retry attempts
        wait_seconds: Seconds to wait between retries
        
    Returns:
        True if operation succeeded, False otherwise
    """
    import psycopg2
    
    for attempt in range(max_retries):
        try:
            result = operation_func()
            if result:
                logger.info(f"✓ {operation_name} succeeded")
                return True
            else:
                logger.warning(f"✗ {operation_name} returned False")
                return False
                
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection error during {operation_name}: {e}")
                logger.info(f"Retrying {operation_name} (Attempt {attempt + 2}/{max_retries})...")
                time.sleep(wait_seconds)
                continue
            else:
                logger.error(f"Failed {operation_name} after {max_retries} attempts due to connection errors")
                return False
                
        except Exception as e:
            logger.error(f"Database operation failed ({operation_name}): {e}", exc_info=True)
            return False
    
    return False


def resolve_pdf_path(path_str: str, config: Config) -> Optional[Path]:
    """
    Resolve a PDF path from string input.
    
    Handles:
    - Local file paths
    - S3 URLs (s3://bucket/key)
    - HTTPS S3 URLs (https://bucket.s3.amazonaws.com/key)
    
    Args:
        path_str: Path string or S3 URL
        config: Application configuration
        
    Returns:
        Local Path to PDF (downloaded if from S3), or None if invalid
    """
    # Check if it's an S3 URL
    if is_s3_url(path_str):
        logger.info(f"Detected S3 URL: {path_str}")
        try:
            # Download to temp/configured directory
            download_dir = Path(config.s3.download_dir) if config.s3.download_dir else None
            local_path = download_from_s3(path_str, config.s3, download_dir=download_dir)
            logger.info(f"Downloaded S3 file to: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return None
    
    # Local path
    local_path = Path(path_str)
    
    # Handle relative paths
    if not local_path.is_absolute():
        # Try relative to current directory
        if local_path.exists():
            return local_path.resolve()
        # Try relative to pdfs directory
        pdf_in_pdfs = config.pdfs_dir / local_path
        if pdf_in_pdfs.exists():
            return pdf_in_pdfs
    
    if local_path.exists():
        return local_path.resolve()
    
    logger.warning(f"File not found: {path_str}")
    return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Electoral Roll PDF Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "paths",
        nargs="*",
        type=str,
        help="PDF files or S3 URLs to process. Supports local paths and s3://bucket/key URLs. "
             "If not specified, processes all PDFs in pdfs/ directory.",
    )
    
    parser.add_argument(
        "--step",
        choices=["extract", "metadata", "crop", "field-crop", "id-crop", "header", "merge", "top-merge", "id-merge", "id-extract", "ocr", "csv", "all"],
        default="all",
        help="Run specific processing step (default: all). 'id-crop' crops ID fields (serial, epic, house) and stitches them. 'id-merge' merges these ID crops into batches.",
    )
    
    parser.add_argument(
        "--folder",
        type=str,
        default="",
        help="Process only a specific extracted folder (for crop/ocr steps)",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available extracted folders and exit",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output exists",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit processing to first N items (0 = all)",
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="DPI for PDF rendering (default: 200 or RENDER_DPI env var)",
    )
    
    parser.add_argument(
        "--languages",
        type=str,
        default="eng+tam",
        help="OCR languages (default: eng+tam)",
    )
    
    parser.add_argument(
        "--diagram-filter",
        choices=["auto", "on", "off"],
        default="auto",
        help="Diagram/photo filter for cropping (default: auto)",
    )
        
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip AI metadata extraction",
    )
    
    parser.add_argument(
        "--dump-raw-ocr",
        action="store_true",
        help="Dump raw OCR text for debugging",
    )
    
    # OCR mode flags (merged + tesseract are now defaults)
    parser.add_argument(
        "--use-crops",
        action="store_true",
        dest="use_crops",
        help="Use individual crop images instead of merged batches (slower but may be more accurate).",
    )
    
    parser.add_argument(
        "--use-tamil-ocr",
        action="store_true",
        dest="use_tamil_ocr",
        help="Use Tamil OCR (ocr_tamil) instead of Tesseract (requires GPU for best performance).",
    )
    
    parser.add_argument(
        "--use-ai-ocr",
        action="store_true",
        help="Use AI (Groq) Vision for OCR extraction instead of local OCR.",
    )

    
    parser.add_argument(
        "--no-csv",
        action="store_false",
        dest="csv",
        help="Disable ID data export to CSV after processing (default: csv export is enabled).",
    )
    
    parser.add_argument(
        "--s3-input",
        nargs="*",
        help="List of S3 PDF URLs to process. Files will be downloaded first.",
    )
    
    parser.add_argument(
        "--s3-output",
        type=str,
        help="S3 directory URL to upload generated CSV files to.",
    )
    
    parser.add_argument(
        "--output-identifier",
        type=str,
        help="Identifier for the output directory structure and CSV field.",
    )

    parser.set_defaults(csv=True)
    
    return parser.parse_args()


def list_extracted_folders(config: Config) -> None:
    """List all extracted folders."""
    logger.info("Extracted folders:")
    
    store = JSONStore(config.extracted_dir)
    
    for folder in iter_extracted_folders(config.extracted_dir):
        name = folder.name
        has_metadata = store.metadata_exists(name)
        has_output = store.document_exists(name)
        
        # Count images and crops
        images_dir = folder / "images"
        crops_dir = folder / "crops"
        
        images_count = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
        crops_count = sum(1 for _ in crops_dir.rglob("*.png")) if crops_dir.exists() else 0
        
        status_parts = []
        if has_metadata:
            status_parts.append("metadata")
        if has_output:
            status_parts.append("processed")
        
        status = ", ".join(status_parts) if status_parts else "extracted only"
        
        logger.info(f"  {name}: {images_count} pages, {crops_count} crops [{status}]")


def export_to_destination(csv_paths: List[Path], destination: str, config: Config) -> None:
    """
    Export CSV files to a secondary destination (S3 or local directory).
    
    Args:
        csv_paths: List of CSV file paths
        destination: Target destination (S3 URL or local path)
        config: Application configuration
    """
    if not csv_paths or not destination:
        return

    # Check if S3
    if is_s3_url(destination):
        logger.info(f"Uploading CSVs to S3: {destination}")
        for csv_path in csv_paths:
            # Construct S3 URL
            folder_url = destination.rstrip("/")
            s3_url = f"{folder_url}/{csv_path.name}"
            
            try:
                upload_to_s3(csv_path, s3_url, config.s3)
                logger.info(f"Uploaded {csv_path.name} to {s3_url}")
            except Exception as e:
                logger.error(f"Failed to upload {csv_path.name} to S3: {e}")
    else:
        # Local directory
        dest_dir = Path(destination)
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying CSVs to local directory: {dest_dir}")
            
            for csv_path in csv_paths:
                dest_file = dest_dir / csv_path.name
                shutil.copy2(csv_path, dest_file)
                logger.info(f"Copied {csv_path.name} to {dest_file}")
        except Exception as e:
            logger.error(f"Failed to copy to local directory {dest_dir}: {e}")


def process_pdf(
    pdf_path: Path,
    config: Config,
    args: argparse.Namespace,
) -> Optional[ProcessedDocument]:
    """
    Process a single PDF through the complete pipeline.
    
    Args:
        pdf_path: Path to PDF file
        config: Configuration
        args: Command line arguments
    
    Returns:
        ProcessedDocument if successful, None otherwise
    """
    timer = Timer()
    
    logger.info(f"Processing: {pdf_path.name}")
    
    # Create processing context
    context = ProcessingContext(config=config)
    context.setup_paths_from_pdf(pdf_path)
    
    # Initialize persistence
    store = JSONStore(config.extracted_dir)
    
    # Create document
    document = ProcessedDocument(
        id=context.pdf_name,
        pdf_name=context.pdf_name,
        pdf_path=str(pdf_path),
    )
    
    # Step 1: Extract images from PDF
    if args.step in ["extract", "all"]:
        logger.info("Step 1: Extracting images from PDF...")
        
        # Determine DPI
        dpi = args.dpi if args.dpi is not None else config.render_dpi
        
        extractor = PDFExtractor(context, dpi=dpi)
        if not extractor.run():
            document.status = "failed"
            document.error = "PDF extraction failed"
            return document
        
        context.total_pages = extractor.result.total_pages
        document.pages_count = context.total_pages
    
    # Step 2: Extract metadata using AI
    if args.step in ["metadata", "all"] and not args.skip_metadata:
        logger.info("Step 2: Extracting metadata using AI...")
        
        metadata_extractor = MetadataExtractor(context, force=True, output_identifier=args.output_identifier)
        if metadata_extractor.run() and metadata_extractor.result:
            document.metadata = metadata_extractor.result
            
            # Critical Check: Language Detection
            if not document.metadata.language_detected:
                logger.error("CRITICAL: Language not detected in metadata. Terminating processing for this file.")
                document.status = "failed"
                document.error = "Language not detected"
                return document

            # Transfer AI usage from context to document stats
            if context.ai_usage:
                document.stats.ai_usage = context.ai_usage
            store.save_metadata(context.pdf_name, metadata_extractor.result)
        else:
            logger.error("CRITICAL: Metadata extraction failed. Terminating processing for this file.")
            document.status = "failed"
            document.error = "Metadata extraction failed"
            return document
    
    # Step 3: Crop voter boxes
    if args.step in ["crop", "all"]:
        logger.info("Step 3: Cropping voter boxes...")
        
        cropper = ImageCropper(context, diagram_filter=args.diagram_filter)
        if not cropper.run():
            logger.warning("Cropping failed or no crops found")
        elif cropper.summary:
            logger.info(f"Cropped {cropper.summary.total_crops} voter boxes")
    
    # Step 3.5: Extract page header metadata
    header_data = {}
    if args.step in ["header", "crop", "ocr", "all"]:
        logger.info("Step 3.5: Extracting page header metadata...")
        
        header_extractor = HeaderExtractor(context, languages=args.languages)
        if header_extractor.run():
            header_data = header_extractor.get_all_headers()
            logger.info(f"Extracted headers from {len(header_data)} pages")
        else:
            logger.warning("Header extraction failed or no headers found")
    
    # Step 3.6: Merge crop-top images with filename labels
    if args.step == "top-merge":
        logger.info("Step 3.6: Merging crop-top images...")
        
        top_merger = CropTopMerger(context)
        if top_merger.run() and top_merger.summary:
            logger.info(
                f"Merged {top_merger.summary.total_images} crop-top images into "
                f"{top_merger.summary.total_batch_files} batches"
            )

    # Step: ID Field Cropping
    if args.step in ["id-crop", "all"]:
        logger.info("Step: ID Field Cropping...")
        id_cropper = IdFieldCropper(context)
        if id_cropper.run() and id_cropper.summary:
             logger.info(
                f"ID cropped {id_cropper.summary.successful_crops} images "
                f"({id_cropper.summary.failed_crops} failed)"
            )
            
    # Step: ID Field Merging
    if args.step in ["id-merge", "all"]:
        logger.info("Step: ID Field Merging...")
        id_merger = IdFieldMerger(context)
        if id_merger.run() and id_merger.summary:
            logger.info(
                f"Merged {id_merger.summary.total_images_merged} ID crops into "
                f"{id_merger.summary.total_batch_files} batches"
            )
    
    # Step: AI ID Extraction
    ai_id_processor = None
    if args.step in ["id-extract", "all"]:
        logger.info("Step: AI ID (Serial/EPIC/House) Extraction...")
        from src.processors import AIIdProcessor
        ai_id_processor = AIIdProcessor(context)
        if ai_id_processor.run():
            logger.info("AI ID extraction complete")
            # Keep the processor for OCR step integration
    elif args.step == "ocr":
        # For OCR-only runs, try to load existing AI ID data from debug files
        from src.processors import AIIdProcessor
        debug_dir = context.output_dir / "debug"
        if debug_dir.exists():
            id_extract_files = list(debug_dir.glob("id_extract_page-*.json"))
            if id_extract_files:
                logger.info(f"Loading existing AI ID data from {len(id_extract_files)} debug files...")
                ai_id_processor = AIIdProcessor(context)
                # Load the data into the processor's page_results
                import json
                from src.processors.ai_id_processor import IdExtractionResult
                for file_path in sorted(id_extract_files):
                    page_id = file_path.stem.replace("id_extract_", "")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        results = [
                            IdExtractionResult(
                                serial_no=item.get("serial_no", ""),
                                house_no=item.get("house_no", "")
                            )
                            for item in data
                        ]
                        ai_id_processor.page_results[page_id] = results
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
                total_loaded = sum(len(v) for v in ai_id_processor.page_results.values())
                logger.info(f"Loaded {total_loaded} AI ID records from {len(ai_id_processor.page_results)} pages")
    
    # Step 4: Merge cropped images
    if args.step in ["merge", "all"]:
        logger.info("Step 4: Merging cropped images...")
        
        merger = ImageMerger(context)
        if merger.run() and merger.summary:
            logger.info(
                f"Merged {merger.summary.total_images_merged} images into "
                f"{merger.summary.total_batch_files} batches"
            )
    
    # Step 5: OCR extraction
    if args.step in ["ocr", "all"]:
        logger.info("Step 5: Running OCR extraction...")
        
        # Define callback for immediate page saving
        def save_page_callback(page_id: str, page_voters: list, page_time: float):
            page_path = store.save_page(
                context.pdf_name, 
                page_id, 
                page_voters,
                page_processing_seconds=page_time
            )
            logger.info(f"Saved page {page_id}: {len(page_voters)} voters in {page_time:.2f}s")
        
        if args.use_ai_ocr:
            from src.processors import AIOCRProcessor
            logger.info("Using AI (Groq) for OCR extraction")
            processor = AIOCRProcessor(
                context,
                on_page_complete=save_page_callback,
            )
        else:
            processor = OCRProcessor(
                context,
                languages=args.languages,
                dump_raw_ocr=args.dump_raw_ocr,
                use_merged=not getattr(args, 'use_crops', False),  # Default: True (merged)
                use_tesseract=not getattr(args, 'use_tamil_ocr', False),  # Default: True (tesseract)
                on_page_complete=save_page_callback,
                ai_id_processor=ai_id_processor,
            )

        
        if processor.run():
            voters = processor.get_all_voters()

            document.add_voters(voters, header_data=header_data)
            logger.info(f"Extracted {len(voters)} voter records")
        else:
            logger.warning("OCR processing failed")
    
    # Finalize
    document.status = "completed"
    document.stats.total_time_sec = timer.elapsed
    
    # Save combined output
    if args.step in ["ocr", "all"]:
        output_path = store.save_document(document)
        logger.info(f"Saved output: {output_path}")
        
        # Also update metadata sidecar with final counts
        if document.metadata:
            store.save_metadata(context.pdf_name, document.metadata, total_voters_extracted=document.total_voters)
        
        # Save to PostgreSQL if configured
        if config.db.is_configured:
            logger.info("Saving to PostgreSQL database...")
            
            def save_to_db():
                from src.persistence.postgres import PostgresRepository
                db_repo = PostgresRepository(config.db)
                db_repo.init_db()
                return db_repo.save_document(document)
            
            retry_database_operation(
                save_to_db,
                f"Save document {document.pdf_name} to database"
            )
        
        
    # Step 5.5: Extract Missing House Numbers
    if args.csv or args.step == "csv":
        if document.status == "completed" or args.step == "csv":
            # Check if there are any missing house numbers and extract them using AI
            if document.pages:
                logger.info("Step 5.5: Checking for missing house numbers...")
                from src.processors import MissingHouseNumberProcessor
                
                missing_house_proc = MissingHouseNumberProcessor(context, document)
                if missing_house_proc.run():
                    if missing_house_proc.result:
                        result = missing_house_proc.result
                        if result.missing_house_count > 0:
                            logger.info(
                                f"Missing house numbers: found={result.missing_house_count}, "
                                f"extracted={result.extracted_count}, failed={result.failed_count}"
                            )
                        else:
                            logger.info("No missing house numbers found")
                    
                    # Update the saved document with corrected house numbers
                    if result.extracted_count > 0:
                        output_path = store.save_document(document)
                        logger.info(f"Updated document with extracted house numbers: {output_path}")

            # Step 5.6: Reprocess Missing Names
            # Run this whenever we are checking for missing data (before CSV export)
            if document.pages:
                logger.info("Step 5.6: Checking for missing voter names and relations...")
                from src.processors import MissingNameProcessor
                
                missing_name_proc = MissingNameProcessor(context, document)
                if missing_name_proc.run():
                    if missing_name_proc.result:
                        result = missing_name_proc.result
                        if result.missing_name_count > 0:
                            logger.info(
                                f"Missing names: found={result.missing_name_count}, "
                                f"OCR-fixed={result.ocr_recovered_count}, "
                                f"AI-fixed={result.ai_recovered_count}"
                            )
                        else:
                            logger.info("No missing voter names found")
                    
                    # Update saved document if any recoveries happened
                    if missing_name_proc.result and (missing_name_proc.result.ocr_recovered_count > 0 or missing_name_proc.result.ai_recovered_count > 0):
                        output_path = store.save_document(document)
                        logger.info(f"Updated document with recovered names: {output_path}")
        
    # Step 5.5: Fix Invalid EPICs before CSV Export
    if args.csv or args.step == "csv":
        if document.status == "completed" or args.step == "csv":
            logger.info("Step 5.5: Fixing invalid EPICs with AI...")
            from src.processors.epic_fixer import EPICFixerProcessor
            
            epic_fixer = EPICFixerProcessor(context)
            if epic_fixer.process_document(document):
                # Save updated document if any EPICs were fixed
                if epic_fixer.fixed_count > 0:
                    output_path = store.save_document(document)
                    logger.info(f"Updated document with {epic_fixer.fixed_count} fixed EPICs: {output_path}")
            else:
                logger.info("EPIC fixer not available or skipped")
    
    # Step 6: CSV Export
    if args.csv or args.step == "csv":
        if document.status == "completed" or args.step == "csv":
            # If step is just CSV, we might need to load existing document if not in memory
            # But process_pdf creates a new document. 
            # If step=csv, we should check if we can load existing data
            if args.step == "csv" and not document.pages:
                existing_data = store.load_document(context.pdf_name)
                if existing_data:
                    logger.info("Loaded existing data for CSV export")
                    store.save_to_csv(existing_data)
                    logger.info(f"Exported CSVs to {context.pdf_name}/output/csv/")
                    return document
                else:
                    logger.warning("No existing processed data found for CSV export")
                    return document
            
            # If we just finished processing (document is populated)
            logger.info("Step 6: Exporting to CSV...")
            csv_paths = store.save_to_csv(document, output_identifier=args.output_identifier)
            logger.info(f"Exported CSVs to {context.pdf_name}/output/csv/")
            
            # S3 Upload / Local Export
            if args.s3_output and csv_paths:
                export_to_destination(csv_paths, args.s3_output, config)
                
            # Update Database after CSV (ensures output_identifier and any late updates are saved)
            if config.db.is_configured and document.status == "completed":
                logger.info("Updating PostgreSQL database after CSV generation...")
                
                def update_db():
                    from src.persistence.postgres import PostgresRepository
                    db_repo = PostgresRepository(config.db)
                    # Ensure output_identifier is in metadata if provided
                    if args.output_identifier and document.metadata:
                        document.metadata.output_identifier = args.output_identifier
                    return db_repo.save_document(document)
                
                retry_database_operation(
                    update_db,
                    f"Update document {document.pdf_name} in database after CSV"
                )
        
    
    return document


def process_extracted_folder(
    folder: Path,
    config: Config,
    args: argparse.Namespace,
) -> Optional[ProcessedDocument]:
    """
    Process an already-extracted folder (for crop/ocr steps).
    
    Args:
        folder: Path to extracted folder
        config: Configuration
        args: Command line arguments
    
    Returns:
        ProcessedDocument if successful, None otherwise
    """
    timer = Timer()
    
    logger.info(f"Processing folder: {folder.name}")
    
    # Create processing context
    context = ProcessingContext(config=config)
    context.setup_paths_from_extracted(folder)
    
    # Initialize persistence
    store = JSONStore(config.extracted_dir)
    
    # Create document
    document = ProcessedDocument(
        id=context.pdf_name,
        pdf_name=context.pdf_name,
    )
    
    # Load existing metadata if available
    existing_metadata = store.load_metadata(context.pdf_name)
    if existing_metadata:
        from src.models import DocumentMetadata, AIUsage
        # Extract AI metadata from the loaded JSON
        ai_meta = existing_metadata.get("ai_metadata", {})
        document.metadata = DocumentMetadata.from_ai_response(
            existing_metadata,
            ai_provider=ai_meta.get("provider", ""),
            ai_model=ai_meta.get("model", ""),
            input_tokens=ai_meta.get("input_tokens", 0),
            output_tokens=ai_meta.get("output_tokens", 0),
            cost_usd=ai_meta.get("cost_usd"),
            extraction_time_sec=ai_meta.get("extraction_time_sec", 0.0),
        )
        # Also populate document stats AI usage
        if ai_meta.get("provider") or ai_meta.get("model"):
            document.stats.ai_usage = AIUsage(
                provider=ai_meta.get("provider", ""),
                model=ai_meta.get("model", ""),
            )
            document.stats.ai_usage.add_call(
                input_tokens=ai_meta.get("input_tokens", 0),
                output_tokens=ai_meta.get("output_tokens", 0),
                cost_usd=ai_meta.get("cost_usd"),
            )
    
    # Step: Crop voter boxes
    if args.step in ["crop", "all"]:
        logger.info("Cropping voter boxes...")
        
        cropper = ImageCropper(context, diagram_filter=args.diagram_filter)
        if cropper.run() and cropper.summary:
            logger.info(f"Cropped {cropper.summary.total_crops} voter boxes")
    
    # Step: Extract page header metadata
    header_data = {}
    if args.step in ["header", "crop", "ocr", "all"]:
        logger.info("Extracting page header metadata...")
        
        header_extractor = HeaderExtractor(context, languages=args.languages)
        if header_extractor.run():
            header_data = header_extractor.get_all_headers()
            logger.info(f"Extracted headers from {len(header_data)} pages")
        else:
            logger.warning("Header extraction failed or no headers found")
    
    # Step: Merge crop-top images with filename labels
    if args.step == "top-merge":
        logger.info("Merging crop-top images...")
        
        top_merger = CropTopMerger(context)
        if top_merger.run() and top_merger.summary:
            logger.info(
                f"Merged {top_merger.summary.total_images} crop-top images into "
                f"{top_merger.summary.total_batch_files} batches"
            )

    # Step: ID Field Cropping
    if args.step in ["id-crop", "all"]:
        logger.info("ID Field Cropping...")
        id_cropper = IdFieldCropper(context)
        if id_cropper.run() and id_cropper.summary:
             logger.info(
                f"ID cropped {id_cropper.summary.successful_crops} images "
                f"({id_cropper.summary.failed_crops} failed)"
            )
            
    # Step: ID Field Merging
    if args.step in ["id-merge", "all"]:
        logger.info("ID Field Merging...")
        id_merger = IdFieldMerger(context)
        if id_merger.run() and id_merger.summary:
            logger.info(
                f"Merged {id_merger.summary.total_images_merged} ID crops into "
                f"{id_merger.summary.total_batch_files} batches"
            )
    
    # Step: AI ID Extraction
    ai_id_processor = None
    if args.step in ["id-extract", "all"]:
        logger.info("AI ID (Serial/EPIC/House) Extraction...")
        from src.processors import AIIdProcessor
        ai_id_processor = AIIdProcessor(context)
        if ai_id_processor.run():
            logger.info("AI ID extraction complete")
    elif args.step == "ocr":
        # For OCR-only runs, try to load existing AI ID data from debug files
        from src.processors import AIIdProcessor
        debug_dir = context.output_dir / "debug"
        if debug_dir.exists():
            id_extract_files = list(debug_dir.glob("id_extract_page-*.json"))
            if id_extract_files:
                logger.info(f"Loading existing AI ID data from {len(id_extract_files)} debug files...")
                ai_id_processor = AIIdProcessor(context)
                # Load the data into the processor's page_results
                import json
                from src.processors.ai_id_processor import IdExtractionResult
                for file_path in sorted(id_extract_files):
                    page_id = file_path.stem.replace("id_extract_", "")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        results = [
                            IdExtractionResult(
                                serial_no=item.get("serial_no", ""),
                                house_no=item.get("house_no", "")
                            )
                            for item in data
                        ]
                        ai_id_processor.page_results[page_id] = results
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
                total_loaded = sum(len(v) for v in ai_id_processor.page_results.values())
                logger.info(f"Loaded {total_loaded} AI ID records from {len(ai_id_processor.page_results)} pages")
    
    # Step: Merge cropped images
    if args.step in ["merge", "all"]:
        logger.info("Merging cropped images...")
        
        merger = ImageMerger(context)
        if merger.run() and merger.summary:
            logger.info(
                f"Merged {merger.summary.total_images_merged} images into "
                f"{merger.summary.total_batch_files} batches"
            )
    
    # Step: OCR extraction
    if args.step in ["ocr", "all"]:
        logger.info("Running OCR extraction...")
        
        # Define callback for immediate page saving
        def save_page_callback(page_id: str, page_voters: list, page_time: float):
            page_path = store.save_page(
                context.pdf_name, 
                page_id, 
                page_voters,
                page_processing_seconds=page_time
            )
            logger.info(f"Saved page {page_id}: {len(page_voters)} voters in {page_time:.2f}s")
        
        if args.use_ai_ocr:
            from src.processors import AIOCRProcessor
            logger.info("Using AI (Groq) for OCR extraction")
            processor = AIOCRProcessor(
                context,
                on_page_complete=save_page_callback,
            )
        else:
            processor = OCRProcessor(
                context,
                languages=args.languages,
                dump_raw_ocr=args.dump_raw_ocr,
                use_merged=not getattr(args, 'use_crops', False),  # Default: True (merged)
                use_tesseract=not getattr(args, 'use_tamil_ocr', False),  # Default: True (tesseract)
                on_page_complete=save_page_callback,
                ai_id_processor=ai_id_processor,
            )

        
        if processor.run():
            voters = processor.get_all_voters()

            document.add_voters(voters, header_data=header_data)
            logger.info(f"Extracted {len(voters)} voter records")
    
    # Finalize
    document.status = "completed"
    document.stats.total_time_sec = timer.elapsed
    
    # Save combined output (only if OCR was run, as crop-only has no voter data)
    if args.step in ["ocr", "all"]:
        output_path = store.save_document(document)
        logger.info(f"Saved output: {output_path}")

        # Also update metadata sidecar with final counts
        if document.metadata:
            store.save_metadata(context.pdf_name, document.metadata, total_voters_extracted=document.total_voters)

        # Save to PostgreSQL if configured
        if config.db.is_configured:
            logger.info("Saving to PostgreSQL database...")
            
            def save_to_db():
                from src.persistence.postgres import PostgresRepository
                db_repo = PostgresRepository(config.db)
                db_repo.init_db()
                return db_repo.save_document(document)
            
            retry_database_operation(
                save_to_db,
                f"Save extracted folder {document.pdf_name} to database"
            )


    # Step: Missing House Numbers Extraction
    if args.csv or args.step == "csv":
        # If we just ran OCR, document is populated
        if args.step in ["ocr", "all"] and document.status == "completed":
            # Check if there are any missing house numbers and extract them using AI
            if document.pages:
                logger.info("Checking for missing house numbers...")
                from src.processors import MissingHouseNumberProcessor
                
                missing_house_proc = MissingHouseNumberProcessor(context, document)
                if missing_house_proc.run():
                    if missing_house_proc.result:
                        result = missing_house_proc.result
                        if result.missing_house_count > 0:
                            logger.info(
                                f"Missing house numbers: found={result.missing_house_count}, "
                                f"extracted={result.extracted_count}, failed={result.failed_count}"
                            )
                        else:
                            logger.info("No missing house numbers found")
                    
                    # Update the saved document with corrected house numbers
                    if result.extracted_count > 0:
                        output_path = store.save_document(document)
                        logger.info(f"Updated document with extracted house numbers: {output_path}")

    # Step: CSV Export
    if args.csv or args.step == "csv":
        # If we just ran OCR, document is populated
        if args.step in ["ocr", "all"] and document.status == "completed":
            logger.info("Exporting to CSV...")
            csv_paths = store.save_to_csv(document, output_identifier=args.output_identifier)
            logger.info(f"Exported CSVs to {context.pdf_name}/output/csv/")
            
            # S3 Upload / Local Export
            if args.s3_output and csv_paths:
                 export_to_destination(csv_paths, args.s3_output, config)

            # Update Database after CSV
            if config.db.is_configured and document.status == "completed":
                logger.info("Updating PostgreSQL database after CSV generation...")
                
                def update_db():
                    from src.persistence.postgres import PostgresRepository
                    db_repo = PostgresRepository(config.db)
                    # Ensure output_identifier is in metadata if provided
                    if args.output_identifier and document.metadata:
                        document.metadata.output_identifier = args.output_identifier
                    return db_repo.save_document(document)
                
                retry_database_operation(
                    update_db,
                    f"Update extracted folder {document.pdf_name} in database after CSV"
                )


        # If we are ONLY running CSV step on existing folder
        elif args.step == "csv":
            logger.info("Loading existing data for CSV export...")
            existing_data = store.load_document(context.pdf_name)
            if existing_data:
                # Fix invalid EPICs before exporting
                logger.info("Step 5.5: Fixing invalid EPICs with AI...")
                from src.processors.epic_fixer import EPICFixerProcessor
                
                epic_fixer = EPICFixerProcessor(context)
                if epic_fixer.process_document(existing_data, pdf_name=context.pdf_name):
                    # Save updated document if any EPICs were fixed
                    if epic_fixer.fixed_count > 0:
                        # Save the updated dict back to JSON file
                        import json
                        output_dir = folder / "output"
                        doc_file = output_dir / f"{context.pdf_name}.json"
                        with open(doc_file, 'w', encoding='utf-8') as f:
                            json.dump(existing_data, f, indent=2, ensure_ascii=False)
                        logger.info(f"Updated document with {epic_fixer.fixed_count} fixed EPICs: {doc_file}")
                else:
                    logger.info("EPIC fixer not available or skipped")
                
                # For CSV-only mode, we export the existing data as-is
                # Missing house number extraction requires full document context and is only done during processing
                logger.info("Exporting existing processed data to CSV...")
                csv_paths = store.save_to_csv(existing_data, output_identifier=args.output_identifier)
                logger.info(f"Exported CSVs to {context.pdf_name}/output/csv/")
                
                # S3 Upload / Local Export
                if args.s3_output and csv_paths:
                    export_to_destination(csv_paths, args.s3_output, config)
            else:
                logger.warning(f"No processed output found for {context.pdf_name}, cannot export CSV")
    
    return document


def process_metadata_only(
    folder: Path,
    config: Config,
    args: argparse.Namespace,
) -> Optional[ProcessedDocument]:
    """
    Process metadata extraction only for an already-extracted folder.
    
    Args:
        folder: Path to extracted folder
        config: Configuration
        args: Command line arguments
    
    Returns:
        ProcessedDocument if successful, None otherwise
    """
    timer = Timer()
    
    logger.info(f"Processing: {folder.name}")
    
    # Create processing context
    context = ProcessingContext(config=config)
    context.setup_paths_from_extracted(folder)
    
    # Initialize persistence
    store = JSONStore(config.extracted_dir)
    
    # Create document
    document = ProcessedDocument(
        id=context.pdf_name,
        pdf_name=context.pdf_name,
    )
    
    # Extract metadata using AI
    logger.info("Step 2: Extracting metadata using AI...")
    
    metadata_extractor = MetadataExtractor(context, force=True, output_identifier=args.output_identifier)
    if metadata_extractor.run() and metadata_extractor.result:
        document.metadata = metadata_extractor.result
        
        # Critical Check: Language Detection
        if not document.metadata.language_detected:
            logger.error("CRITICAL: Language not detected in metadata. Terminating processing for this file.")
            document.status = "failed"
            document.error = "Language not detected"
            return document

        # Transfer AI usage from context to document stats
        if context.ai_usage:
            document.stats.ai_usage = context.ai_usage
        # NOTE: Do NOT call store.save_metadata here!
        # The metadata_extractor already saved the file with AI usage data
    else:
        logger.error(f"CRITICAL: Metadata extraction failed for {folder.name}")
        document.status = "failed"
        document.error = "Metadata extraction failed"
        return document
            
    # Finalize
    document.status = "completed"
    document.stats.total_time_sec = timer.elapsed
    
    return document


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logger()
    
    # Load configuration
    config = Config()
    
    logger.info("Electoral Roll PDF Processing Pipeline")
    logger.info(f"Debug mode: {'enabled' if config.debug else 'disabled'}")
    
    # Database connection check - optional or mandatory based on DB_REQUIRED flag
    logger.info("=" * 60)
    if config.db.required:
        logger.info("DATABASE CONNECTION CHECK (REQUIRED)")
    else:
        logger.info("DATABASE CONNECTION CHECK (OPTIONAL)")
    logger.info("=" * 60)
    
    if not config.db.is_configured:
        if config.db.required:
            logger.error("✗ Database is NOT configured in .env file")
            logger.error("TERMINATING: Database connection is required (DB_REQUIRED=true)")
            logger.info("")
            logger.info("Please configure database in .env file:")
            logger.info("  DB_HOST=localhost")
            logger.info("  DB_PORT=5432")
            logger.info("  DB_NAME=electoral_rolls")
            logger.info("  DB_USER=postgres")
            logger.info("  DB_PASSWORD=your_password")
            logger.info("=" * 60)
            return 1
        else:
            logger.warning("⚠ Database is NOT configured in .env file")
            logger.warning("DB_REQUIRED=false, continuing without database")
            logger.info("=" * 60)
            # Continue processing without database
    else:
        # Database is configured, verify connection
        logger.info("✓ Database Configuration Found")
        logger.info(f"    Host: {config.db.host}:{config.db.port}")
        logger.info(f"    Database: {config.db.name}")
        logger.info(f"    User: {config.db.user}")
        logger.info(f"    Required: {config.db.required}")
        logger.info("")
        logger.info("Testing database connection...")
        
        try:
            from src.persistence.postgres import PostgresRepository
            db_repo = PostgresRepository(config.db)
            connection_successful = db_repo.test_connection()
            
            if connection_successful:
                logger.info("✓ Database connection VERIFIED successfully")
                logger.info("=" * 60)
            else:
                # Connection failed but DB not required
                logger.warning("⚠ Database connection failed but DB_REQUIRED=false")
                logger.warning("Continuing without database")
                logger.info("=" * 60)
        except Exception as e:
            if config.db.required:
                logger.error(f"✗ Database connection FAILED: {e}")
                logger.error("TERMINATING: Cannot proceed without database connection (DB_REQUIRED=true)")
                logger.info("=" * 60)
                return 1
            else:
                logger.warning(f"⚠ Database connection failed: {e}")
                logger.warning("DB_REQUIRED=false, continuing without database")
                logger.info("=" * 60)
    
    # Handle --list
    if args.list:
        list_extracted_folders(config)
        return 0
    
    # Determine what to process
    total_timer = Timer()
    
    results: List[ProcessedDocument] = []
    
    if args.step in ["crop", "field-crop", "id-crop", "merge", "top-merge", "id-merge", "id-extract", "ocr", "metadata", "csv"] and not args.paths:
        # Process extracted folders for metadata, crop, or ocr steps
        folders = list(iter_extracted_folders(config.extracted_dir))
        
        if args.folder:
            folders = [f for f in folders if f.name == args.folder]
        
        if args.limit > 0:
            folders = folders[:args.limit]
        
        if not folders:
            logger.error("No extracted folders found")
            return 1
        
        logger.info(f"Processing {len(folders)} extracted folder(s)")
        
        for folder in folders:
            # For metadata step, use a special handler
            if args.step == "metadata":
                doc = process_metadata_only(folder, config, args)
            else:
                doc = process_extracted_folder(folder, config, args)
            if doc:
                results.append(doc)
    
    else:
        # Override extracted_dir if output_identifier is provided BEFORE any processing
        if args.output_identifier:
            # Structure: <output_identifier>/extracted/
            new_extracted_dir = config.base_dir / args.output_identifier / "extracted"
            logger.info(f"Using output identifier directory: {new_extracted_dir}")
            config.extracted_dir = new_extracted_dir
            config.extracted_dir.mkdir(parents=True, exist_ok=True)
            
        # Process PDFs (local or S3)
        pdfs: List[Path] = []
        
        # 1. Handle explicit --s3-input
        if args.s3_input:
            logger.info("Processing S3 input files...")
            
            # Handle potential comma-separated values in the input list
            all_s3_urls = []
            for item in args.s3_input:
                for url in item.split(','):
                    url = url.strip()
                    if url:
                        all_s3_urls.append(url)
            
            for s3_url in all_s3_urls:
                try:
                    # Download to default S3 download dir or temp
                    dl_path = download_from_s3(s3_url, config.s3)
                    if dl_path and dl_path.exists():
                        pdfs.append(dl_path)
                        logger.info(f"Added S3 file: {dl_path.name}")
                    else:
                        logger.error(f"Failed to download or locate S3 file: {s3_url}")
                except Exception as e:
                    logger.error(f"Error processing S3 input {s3_url}: {e}")

        # 2. Handle positional args (paths)
        if args.paths:
            # Resolve each path (handles S3 URLs and local paths)
            for path_str in args.paths:
                resolved = resolve_pdf_path(path_str, config)
                if resolved:
                    pdfs.append(resolved)
                else:
                    logger.warning(f"Skipping invalid path: {path_str}")
        
        # If no paths specified anywhere, default to pdfs dir
        if not pdfs and not args.s3_input and not args.paths:
             # Default: process all PDFs in pdfs directory
            pdfs = list(iter_pdfs(config.pdfs_dir))
        
        if args.limit > 0:
            pdfs = pdfs[:args.limit]
        
        if not pdfs:
            logger.error("No PDF files found")
            return 1
        
        logger.info(f"Processing {len(pdfs)} PDF(s)")
        
        for pdf_path in pdfs:
            doc = process_pdf(pdf_path, config, args)
            if doc:
                results.append(doc)
    
    # Summary
    total_time = total_timer.elapsed
    
    successful = len([r for r in results if r.status == "completed"])
    failed = len([r for r in results if r.status == "failed"])
    total_voters = sum(r.total_voters for r in results)
    
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Documents processed: {successful}/{len(results)}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total voters extracted: {total_voters}")
    logger.info(f"Total time: {total_time:.2f}s")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
