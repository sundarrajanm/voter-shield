import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pdf2image import convert_from_path

from logger import setup_logger

logger = setup_logger()

START_PAGE = 3  # skip first 2 pages


def _convert_single_pdf(pdf_path: str, jpg_dir: str, dpi: int):
    logger.info(f"🚀 Converting {pdf_path}")
    pages = convert_from_path(
        pdf_path,
        dpi=dpi,
        fmt="jpeg",
        thread_count=4,
        jpegopt={"quality": 95},
        first_page=START_PAGE,
    )

    total_pages = len(pages) - 1  # skip last page

    logger.info(f"📝 {pdf_path}: {total_pages} pages to convert")
    for i, page in enumerate(pages):
        if i < total_pages:
            out = os.path.join(
                jpg_dir, f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{i+1:02d}.jpg"
            )
            page.save(out, "JPEG")
            logger.info(f"   ✅ Saved {out}")
    logger.info(f"✅ Completed conversion of {pdf_path}")


def convert_pdfs_to_jpg(
    pdf_dir: str, jpg_dir: str, dpi: int, progress=None, max_workers=4, limit=None
):
    start_time = time.perf_counter()

    os.makedirs(jpg_dir, exist_ok=True)

    pdf_files = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))

    if limit is not None:
        pdf_files = pdf_files[:limit]

    if not pdf_files:
        logger.warning("No PDFs found")
        return

    logger.info(f"🔄 Starting conversion of {len(pdf_files)} PDFs using {max_workers} threads")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {
            executor.submit(_convert_single_pdf, os.path.join(pdf_dir, file), jpg_dir, dpi): file
            for file in pdf_files
        }

        completed = 0
        for future in as_completed(future_to_pdf):
            future.result()
            completed += 1

    elapsed = time.perf_counter() - start_time
    logger.info(f"⏱️ PDF → JPG completed in {elapsed:.2f} seconds")
