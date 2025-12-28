# main.py

import argparse
import json
import os
import time

from rich.console import Console

from config import CROPS_DIR, CSV_DIR, DPI, JPG_DIR, OCR_DIR, PDF_DIR
from crop_voters import crop_voter_boxes_parallel
from csv_extract import clean_and_extract_csv_v2
from logger import setup_logger
from ocr_extract import assign_serial_numbers, extract_voters_from_stacked_txt_files
from pdf_to_png import convert_pdfs_to_jpg
from progress import get_progress
from s3_helper import download_pdfs, upload_directory
from write_csv import write_final_csv

console = Console(force_terminal=True)
logger = setup_logger()

max_workers = 1

def clean_directory(dir_path: str):
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)


def main():
    parser = argparse.ArgumentParser(description="VoterShield Pipeline")

    parser.add_argument("--delete-old", action="store_true")
    parser.add_argument("--regression", action="store_true")

    parser.add_argument(
        "--s3-input",
        help="Comma-separated list of s3:// paths to input PDFs",
    )
    parser.add_argument(
        "--s3-output",
        help="s3:// path where output CSV should be uploaded",
    )

    args = parser.parse_args()

    logger.info(f"🛡️ VoterShield Pipeline Started using {max_workers} thread(s)")

    # --- Handle S3 input ---
    if args.s3_input:
        logger.info("📥 S3 input detected, preparing PDF directory")

        clean_directory(PDF_DIR)

        s3_inputs = [p.strip() for p in args.s3_input.split(",") if p.strip()]
        download_pdfs(s3_inputs, PDF_DIR)

    # --- Delete old intermediate files ---
    if args.delete_old:
        for dir_path in [JPG_DIR, CROPS_DIR, OCR_DIR, CSV_DIR]:
            clean_directory(dir_path)

    start_time = time.perf_counter()
    progress = get_progress()

    with progress:
        # 1️⃣ PDF → JPG
        convert_pdfs_to_jpg(
            PDF_DIR if not args.regression else "tests/fixtures",
            JPG_DIR,
            DPI,
            progress=progress,
            max_workers=max_workers,
            limit=None,
        )
        logger.info("✅ PDFs conversion completed")

        # 2️⃣ Crop voter boxes
        total_crops = crop_voter_boxes_parallel(JPG_DIR, progress=progress, max_workers=max_workers)
        logger.info(f"✅ Cropping completed: {len(total_crops)} crops extracted")

        # 3️⃣ OCR extraction
        extract_voters_from_stacked_txt_files(CROPS_DIR, progress=progress, limit=None)
        with open("ocr/ocr_results.json", encoding="utf-8") as f:
            ocr_results = json.load(f)

        logger.info(f"📊 OCR completed — {len(ocr_results)} blocks")

        # 4️⃣ CSV extraction
        cleaned_records = clean_and_extract_csv_v2(ocr_results, progress=progress)
        cleaned_records = assign_serial_numbers(cleaned_records)

        with open("ocr/cleaned_records.json", "w", encoding="utf-8") as f:
            json.dump(cleaned_records, f, ensure_ascii=False, indent=2)

        # 5️⃣ Write CSV
        write_final_csv(cleaned_records, CSV_DIR)
        logger.info("✅ Final CSV written")

    # --- Handle S3 output ---
    if args.s3_output:
        logger.info("📤 Uploading results to S3")
        upload_directory(CSV_DIR, args.s3_output)

    elapsed = time.perf_counter() - start_time
    logger.info(f"🎉 Pipeline completed successfully in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
