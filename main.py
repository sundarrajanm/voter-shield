# main.py

import argparse
import os
import time
from config import (
    PDF_DIR, PNG_DIR, CROPS_DIR, OCR_DIR, CSV_DIR, DPI
)

from pdf_to_png import convert_pdfs_to_png
from crop_voters import crop_voter_boxes_parallel
from ocr_extract import extract_ocr_from_crops_in_parallel, assign_serial_numbers
from csv_extract import clean_and_extract_csv
from write_csv import write_final_csv

from logger import setup_logger
from progress import get_progress

from rich.console import Console

logger = setup_logger()
console = Console()

max_workers=4

def main():
    logger.info("🛡️ VoterShield Pipeline Started")

    progress = get_progress()

    # Delete files based on command line argument --delete-old
    parser = argparse.ArgumentParser(description="VoterShield Pipeline")
    parser.add_argument("--delete-old", action="store_true", help="Delete old files before starting the pipeline")
    args = parser.parse_args()

    DELETE_OLD_FILES = args.delete_old
    if DELETE_OLD_FILES:
        for dir_path in [PNG_DIR, CROPS_DIR, OCR_DIR, CSV_DIR]:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.error(f"❌ Error deleting file {file_path}: {e}")

    # Record the start time
    start_time = time.perf_counter() # Or time.time() for less precision

    with progress:
        # 1️⃣ PDF → PNG
        convert_pdfs_to_png(
            PDF_DIR,
            PNG_DIR,
            DPI,
            progress=progress,
            max_workers=max_workers,
            limit=2
        )
        logger.info("✅ PDFs conversion completed")

        # 2️⃣ Crop voter boxes
        total_crops = crop_voter_boxes_parallel(
            PNG_DIR,
            progress=progress,
            max_workers=max_workers
        )
        logger.info("✅ Cropping completed")

        # 3️⃣ OCR extraction
        ocr_results = extract_ocr_from_crops_in_parallel(
            total_crops,
            progress=progress,
            max_workers=8,
            limit=None
        )
        logger.info(f"📊 OCR completed — {len(ocr_results)} blocks")

        ### Fast in-memory processing below ###        
        # 4️⃣ Assign serial numbers
        ocr_results = assign_serial_numbers(ocr_results)

        # 5️⃣ CSV extraction
        cleaned_records = clean_and_extract_csv(ocr_results, progress=progress)
        logger.info(f"📊 Extracted {len(cleaned_records)} voters")

        # 5️⃣ Write CSV
        task = progress.add_task("💾 Writing final CSV", total=1)
        write_final_csv(cleaned_records, CSV_DIR)
        progress.update(task, advance=1)
        logger.info("✅ Final CSV written")

    logger.info("🎉 Pipeline completed successfully!")

    # Record the end time
    end_time = time.perf_counter() # Or time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    logger.info(f"Total pipeline time: {elapsed_time:.3f} seconds.")

if __name__ == "__main__":
    main()

# import json
# with open("ocr/ocr_results.json", "r", encoding="utf-8") as f:
#     ocr_results = json.load(f)
