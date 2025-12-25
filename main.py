# main.py

import argparse
import os
import time

from rich.console import Console

from config import CROPS_DIR, CSV_DIR, DPI, JPG_DIR, OCR_DIR, PDF_DIR, PNG_DIR
from crop_voters import crop_voter_boxes_parallel
from csv_extract import clean_and_extract_csv_v2
from logger import setup_logger
from ocr_extract import (
    assign_serial_numbers,
    extract_voters_from_stacked_txt_files,
)
from pdf_to_png import convert_pdfs_to_jpg
from progress import get_progress
from write_csv import write_final_csv

console = Console(force_terminal=True)

logger = setup_logger()

max_workers = 4

def main():
    logger.info("🛡️ VoterShield Pipeline Started")

    progress = get_progress()

    # Delete files based on command line argument --delete-old
    parser = argparse.ArgumentParser(description="VoterShield Pipeline")
    parser.add_argument("--delete-old", action="store_true", help="Delete old files before starting the pipeline")
    parser.add_argument("--regression", action="store_true", help="Run in regression test mode with test PDFs")

    args = parser.parse_args()

    DELETE_OLD_FILES = args.delete_old
    if DELETE_OLD_FILES:
        for dir_path in [JPG_DIR, PNG_DIR, CROPS_DIR, OCR_DIR, CSV_DIR]:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.error(f"❌ Error deleting file {file_path}: {e}")

    regression = args.regression

    # Record the start time
    start_time = time.perf_counter() # Or time.time() for less precision

    with progress:
        # 1️⃣ PDF → JPG
        convert_pdfs_to_jpg(
            # pass test folder path if in regression mode
            PDF_DIR if not regression else "tests/fixtures",
            JPG_DIR,
            DPI,
            progress=progress,
            max_workers=max_workers,
            limit=1
        )
        logger.info("✅ PDFs conversion completed")

        # 2️⃣ Crop voter boxes
        total_crops = crop_voter_boxes_parallel(
            JPG_DIR,
            progress=progress,
            max_workers=max_workers
        )
        logger.info(f"✅ Cropping completed: {len(total_crops)} crops extracted")

        # 3️⃣ OCR extraction
        ocr_results = extract_voters_from_stacked_txt_files(
            CROPS_DIR,
            progress=progress
        )
        logger.info(f"📊 OCR completed — {len(ocr_results)} blocks")

        # 4️⃣ CSV extraction
        # Read ocr_results from ocr/ocr_results.json
        import json
        with open("ocr/ocr_results.json", encoding="utf-8") as f:
                ocr_results = json.load(f)
    
        cleaned_records = clean_and_extract_csv_v2(ocr_results, progress=progress)
        logger.info(f"📊 Extracted {len(cleaned_records)} voters")

        # 4️⃣ Assign serial numbers
        cleaned_records = assign_serial_numbers(cleaned_records)

        # Write cleaned_records to JSON for inspection
        with open("ocr/cleaned_records.json", "w", encoding="utf-8") as f:
            json.dump(cleaned_records, f, ensure_ascii=False, indent=2)

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
