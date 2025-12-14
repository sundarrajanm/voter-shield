# main.py

import os
from config import (
    PDF_DIR, PNG_DIR, CROPS_DIR, CSV_DIR, DPI
)

from pdf_to_png import convert_pdfs_to_png
from crop_voters import crop_voter_boxes
from ocr_extract import extract_ocr_from_crops_in_parallel
from csv_extract import clean_and_extract_csv
from write_csv import write_final_csv

from logger import setup_logger
from progress import get_progress

from rich.console import Console

logger = setup_logger()
console = Console()

def main():
    logger.info("🛡️ VoterShield Pipeline Started")

    progress = get_progress()

    with progress:
        # 1️⃣ PDF → PNG
        convert_pdfs_to_png(PDF_DIR, PNG_DIR, DPI, progress=progress)
        logger.info("✅ PDF conversion completed")

        # 2️⃣ Crop voter boxes
        crop_voter_boxes(PNG_DIR, CROPS_DIR, progress=progress)
        logger.info("✅ Cropping completed")

        # 3️⃣ OCR extraction
        logger.info("🔍 Starting OCR extraction")
        ocr_results = extract_ocr_from_crops_in_parallel(
            CROPS_DIR,
            progress=progress
        )
        logger.info(f"📊 OCR completed — {len(ocr_results)} blocks")

        # 4️⃣ CSV extraction
        logger.info("🧠 Parsing OCR → structured voters")
        cleaned_records = clean_and_extract_csv(ocr_results, progress=progress)
        logger.info(f"📊 Extracted {len(cleaned_records)} voters")

        # 5️⃣ Write CSV
        task = progress.add_task("💾 Writing final CSV", total=1)
        write_final_csv(cleaned_records, CSV_DIR)
        progress.update(task, advance=1)
        logger.info("✅ Final CSV written")

    logger.info("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()

# import json
# with open("ocr/ocr_results.json", "r", encoding="utf-8") as f:
#     ocr_results = json.load(f)
