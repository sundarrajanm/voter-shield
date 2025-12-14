import time
from PIL import Image, ImageDraw
import os
import pytesseract
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from logger import setup_logger
logger = setup_logger()

def extract_epic_id(crop):
    cw, ch = crop.size

    # Crop the entire right 40%
    x1 = int(cw * 0.60)   # left boundary for EPIC region
    x2 = cw               # till the rightmost edge
    y1 = 0
    y2 = ch

    epic_region = crop.crop((x1, y1, x2, y2))

    # Run OCR
    epic_text = pytesseract.image_to_string(
        epic_region,
        lang="eng",
        config=(
            "--psm 7 --oem 1 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
    ).strip()

    # Clean output to allow only alphanumeric
    epic_text = "".join(c for c in epic_text if c.isalnum())

    return epic_text

def extract_text_from_image(image_path: str) -> str:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng", config="--psm 6 --oem 1")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

# def extract_ocr_from_crops(crops_dir: str, progress=None, limit=None):
#     """
#     Performs OCR on all cropped voter images.

#     Returns:
#         list[dict]: [
#             {
#                 "source_image": "voter_01.png",
#                 "ocr_text": "Name : ...",
#             },
#             ...
#         ]
#     """
#     files = sorted(os.listdir(crops_dir))
#     task = None

#     if progress:
#         task = progress.add_task("🔍 OCR crops", total=len(files))
    
#     results = []

#     # Record the start time
#     start_time = time.perf_counter() # Or time.time() for less precision

#     serial_no = 1
#     for file in files:
#         if progress:
#             progress.advance(task)

#         if file.lower().endswith(".png"):
#             path = os.path.join(crops_dir, file)

#             ocr_text = extract_text_from_image(path)
#             epic_id = extract_epic_id(Image.open(path))

#             if limit is not None and serial_no > limit:
#                 break

#             if ocr_text.strip() != "":
#                 results.append({
#                     "source_image": file,
#                     "ocr_text": ocr_text,
#                     "epic_id": epic_id,
#                     "serial_no": serial_no,
#                 })
#             serial_no += 1

#     # Write results to ocr_results.json for inspection
#     import json
#     with open("ocr/ocr_results.json", "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     # Record the end time
#     end_time = time.perf_counter() # Or time.time()

#     # Calculate the elapsed time
#     elapsed_time = end_time - start_time

#     logger.info(f"Total number of OCR results: {len(results)}")
    
#     logger.info(f"Time taken by extract_ocr_from_crops: {elapsed_time:.3f} seconds.")
#     return results

# def extract_ocr_from_png(png_dir: str, batch_size):
#     """
#     Processes all voter png images in batches and returns cleaned results.
#     """

#     # 1️⃣ Collect & sort png files
#     png_files = sorted(
#         f for f in os.listdir(png_dir)
#         if f.lower().endswith(".png")
#     )

#     logger.info(f"📁 Found {len(png_files)} png images")

#     all_results = []

#     total_start = time.time()

#     for idx, file in enumerate(png_files, start=1):
#         if idx == batch_size + 1:
#             logger.info("\nReached batch size limit for testing. Stopping further processing.")
#             break  # Limit to first `batch_size` files for testing

#         path = os.path.join(png_dir, file)

#         # 2️⃣ OCR extraction
#         logger.info(f"🔍 OCR processing: {file}")
#         ocr_text = extract_text_from_image(path)
#         all_results.append({
#             "source_image": file,
#             "ocr_text": ocr_text,
#         })

#     # Write results to ocr_results.json for inspection
#     import json
#     with open("ocr/ocr_results.json", "w", encoding="utf-8") as f:
#         json.dump(all_results, f, ensure_ascii=False, indent=2)

#     total_end = time.time()
    
#     logger.info(f"\nTotal time taken by extract_ocr_from_png: {total_end - total_start:.3f} seconds.\nReturning {len(all_results)} results.")

#     logger.info("\n🎉 All OCR processing completed")
    
#     return all_results

import re
from typing import NamedTuple

class ParsedFile(NamedTuple):
    doc_id: str
    page_no: int
    voter_no: int

FILENAME_RE = re.compile(
    r"^(?P<doc>.+?)_page_(?P<page>\d+)_voter_(?P<voter>\d+)\.png$",
    re.IGNORECASE
)

def parse_filename(filename: str) -> ParsedFile | None:
    m = FILENAME_RE.match(filename)
    if not m:
        return None

    return ParsedFile(
        doc_id=m.group("doc"),
        page_no=int(m.group("page")),
        voter_no=int(m.group("voter"))
    )


def _ocr_worker(file, crops_dir):
    """
    Worker function executed in a thread.
    """
    try:
        path = os.path.join(crops_dir, file)

        ocr_text = extract_text_from_image(path)
        epic_id = extract_epic_id(Image.open(path))

        return {
            "source_image": file,
            "ocr_text": ocr_text,
            "epic_id": epic_id,
        }
    except Exception as e:
        logger.error(f"OCR failed for {file}: {e}")
        return {
            "source_image": file,
            "ocr_text": "",
            "epic_id": None,
        }


def extract_ocr_from_crops_in_parallel(crops_dir: str, progress=None, max_workers=4, limit=None):
    """
    Performs OCR on all cropped voter images using multi-threading.
    """
    files = sorted(f for f in os.listdir(crops_dir) if f.lower().endswith(".png"))

    if limit:
        files = files[:limit]

    task = None
    if progress:
        task = progress.add_task("🔍 OCR crops", total=len(files))

    results = []
    start_time = time.perf_counter()

    logger.info(
        f"🔍 Starting OCR extraction ({len(files)} crops, {max_workers} threads)"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_ocr_worker, file, crops_dir): file
            for file in files
        }

        for future in as_completed(futures):
            file = futures[future]
            result = future.result()

            if progress:
                progress.advance(task)

            if not result["ocr_text"].strip():
                continue

            parsed = parse_filename(file)
            if not parsed:
                continue

            result["doc_id"] = parsed.doc_id
            result["page_no"] = parsed.page_no
            result["voter_no"] = parsed.voter_no

            results.append(result)

    # Persist debug output
    with open("ocr/ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed_time = time.perf_counter() - start_time

    logger.info(f"Total number of OCR results: {len(results)}")
    logger.info(
        f"Time taken by extract_ocr_from_crops: {elapsed_time:.3f} seconds."
    )

    return results

from collections import defaultdict

def assign_serial_numbers(results: list[dict]) -> list[dict]:
    """
    Assigns serial_no resetting per doc_id.
    """

    # Group by doc_id
    grouped = defaultdict(list)
    for r in results:
        grouped[r["doc_id"]].append(r)

    final = []

    for doc_id, voters in grouped.items():
        # Sort deterministically inside doc
        voters.sort(key=lambda x: (x["page_no"], x["voter_no"]))

        # Assign serial_no
        for idx, voter in enumerate(voters, start=1):
            voter["serial_no"] = idx
            final.append(voter)

    # Optional: global stable ordering
    final.sort(key=lambda x: (x["doc_id"], x["serial_no"]))
    return final
