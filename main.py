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
from s3_helper import download_pdfs
from logger import setup_logger
from progress import get_progress

from rich.console import Console
console = Console(force_terminal=True)

logger = setup_logger()

max_workers=4
def create_folders(CONSTITUENCY_NO, BATCH_NO):
    BASE_RUN_DIR = os.path.join("tmp", f"ac_{CONSTITUENCY_NO}_batch_{BATCH_NO}")
    PDF_DIR = os.path.join(BASE_RUN_DIR, "pdf")
    PNG_DIR = os.path.join(BASE_RUN_DIR, "png")
    CROPS_DIR = os.path.join(BASE_RUN_DIR, "crops")
    OCR_DIR = os.path.join(BASE_RUN_DIR, "ocr")
    CSV_DIR = os.path.join(BASE_RUN_DIR, "csv")

    for d in ["tmp",BASE_RUN_DIR, PDF_DIR, PNG_DIR, CROPS_DIR, OCR_DIR, CSV_DIR]:
        os.makedirs(d, exist_ok=True)
    logger.info("Folders created Successfully")
def main():
    logger.info("🛡️ VoterShield Pipeline Started")

    progress = get_progress()

    parser = argparse.ArgumentParser(description="VoterShield Pipeline")
    parser.add_argument("--constituency-number", type=int, required=True, help="Constituency number (e.g. 145)")
    parser.add_argument("--batch-no",type=str,required=True,help="Batch identifier (e.g. B01, B02)")
    parser.add_argument("--pdf-files",nargs="+",required=False,help="List of PDF S3 paths for this batch")
    parser.add_argument("--delete-old",action="store_true",help="Delete old files before starting the pipeline")
    parser.add_argument(
        "--resume-from",
        dest="resume_from",
        choices=["pdf2png", "crop", "ocr", "assign_serial", "extract_csv", "write_csv"],
        help=("Resume execution from given stage. Choices: pdf2png, crop, ocr, "
            "assign_serial, extract_csv, write_csv")
    )
    args = parser.parse_args()
    logger.info(f"Request: {args}")
    CONSTITUENCY_NO = args.constituency_number
    BATCH_NO = args.batch_no
    PDF_FILES = args.pdf_files
    RESUME_FROM = args.resume_from
    DELETE_OLD_FILES = args.delete_old

    logger.info(f"🏷️ Constituency={CONSTITUENCY_NO}, Batch={BATCH_NO}, PDFs={PDF_FILES}")
    create_folders(CONSTITUENCY_NO, BATCH_NO)
    if PDF_FILES is not None and len(PDF_FILES)>0:
        logger.info("⬇️ Downloading PDFs from S3")
        download_pdfs(PDF_FILES, PDF_DIR)

    if RESUME_FROM:
        logger.info(f"🔁 Resuming pipeline from stage: {RESUME_FROM}")
        exit
    # Determine which stages to run. If RESUME_FROM is provided, skip stages
    # that come before it in this ordered list.
    _stages = ["pdf2png", "crop", "ocr", "assign_serial", "extract_csv", "write_csv"]
    run_stage = {s: True for s in _stages}
    if RESUME_FROM:
        try:
            start_idx = _stages.index(RESUME_FROM)
            for s in _stages[:start_idx]:
                run_stage[s] = False
        except ValueError:
            # argparse choices should prevent this, but keep safeguard
            logger.warning(f"Unknown resume stage '{RESUME_FROM}'; running full pipeline")

    if DELETE_OLD_FILES:
        logger.info("Deleting the old files")
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
        if run_stage.get("pdf2png", True):
            convert_pdfs_to_png(
                PDF_DIR,
                PNG_DIR,
                DPI,
                progress=progress,
                max_workers=max_workers,
                limit=1
            )
        else:
            logger.info("⏭️ Skipping PDF -> PNG stage (resuming)")
        logger.info("✅ PDFs conversion completed")

        # 2️⃣ Crop voter boxes
        if run_stage.get("crop", True):
            total_crops = crop_voter_boxes_parallel(
                PNG_DIR,
                progress=progress,
                max_workers=max_workers
            )
        else:
            logger.info("⏭️ Skipping crop stage (resuming)")
            # If crop was skipped we still need total_crops for downstream; set to None
            total_crops = None
        logger.info("✅ Cropping completed")

        # 3️⃣ OCR extraction
        if run_stage.get("ocr", True):
            ocr_results = extract_ocr_from_crops_in_parallel(
                total_crops,
                progress=progress,
                max_workers=8,
                limit=None
            )
            logger.info(f"📊 OCR completed — {len(ocr_results)} blocks")
        else:
            logger.info("⏭️ Skipping OCR stage (resuming)")
            ocr_results = []

        ### Fast in-memory processing below ###        
        # 4️⃣ Assign serial numbers
        if run_stage.get("assign_serial", True):
            ocr_results = assign_serial_numbers(ocr_results)
        else:
            logger.info("⏭️ Skipping serial assignment (resuming)")

        # 5️⃣ CSV extraction
        if run_stage.get("extract_csv", True):
            cleaned_records = clean_and_extract_csv(ocr_results, progress=progress)
            logger.info(f"📊 Extracted {len(cleaned_records)} voters")
        else:
            logger.info("⏭️ Skipping CSV extraction (resuming)")
            cleaned_records = []

        # 6️⃣ Write CSV
        if run_stage.get("write_csv", True):
            task = progress.add_task("💾 Writing final CSV", total=1)
            write_final_csv(cleaned_records, CSV_DIR)
            progress.update(task, advance=1)
            logger.info("✅ Final CSV written")
        else:
            logger.info("⏭️ Skipping write CSV stage (resuming)")

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
