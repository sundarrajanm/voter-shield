import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

import pytesseract

from logger import isDebugMode, setup_logger
from utilities import parse_page_metadata_tamil

logger = setup_logger()


def detect_ocr_language_from_filename(filename: str) -> str:
    """
    Detect OCR language based on PNG/PDF filename.

    Returns:
        "eng"      → for English-only OCR
        "tam+eng"  → for Tamil + English OCR
    """
    fname = filename.upper()

    if "-TAM-" in fname:
        return "tam+eng"
    elif "-ENG-" in fname:
        return "eng"
    else:
        # Safe default (numbers + English labels still work)
        return "eng"


def extract_epic_id(crop):
    cw, ch = crop.size

    # Crop the entire right 40%
    x1 = int(cw * 0.60)  # left boundary for EPIC region
    x2 = cw  # till the rightmost edge
    y1 = 0
    y2 = ch

    epic_region = crop.crop((x1, y1, x2, y2))

    # Run OCR
    epic_text = pytesseract.image_to_string(
        epic_region,
        lang="eng",
        config=(
            "--psm 7 --oem 1 " "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ),
    ).strip()

    # Clean output to allow only alphanumeric
    epic_text = "".join(c for c in epic_text if c.isalnum())

    return epic_text


def extract_text_from_image(crop, lang="eng") -> str:
    text = pytesseract.image_to_string(crop, lang=lang, config="--psm 6 --oem 1")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


class ParsedFile(NamedTuple):
    doc_id: str
    page_no: int
    assembly: str
    part_no: int
    street: str


FILENAME_RE = re.compile(r"^(?P<doc>.+?)_page_(?P<page>\d+)_stacked_ocr.txt$", re.IGNORECASE)


def parse_page_metadata(ocr_text: str) -> dict[str, str | None]:
    result = {"assembly": None, "part_no": None, "street": None}
    if not ocr_text:
        return result

    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return result

    line1, line2 = lines[0], lines[1]

    # Assembly: between ':' and 'Part'
    m_assembly = re.search(r"Name\s*:\s*([A-Za-z0-9\- ]+?)\s+Part", line1, re.I)
    if m_assembly:
        result["assembly"] = m_assembly.group(1).strip()

    # Part No
    m_part = re.search(r"Part\s*No\.?\s*[:\-]?\s*(\d+)", line1, re.I)
    if m_part:
        result["part_no"] = int(m_part.group(1))

    # Street: keep the "1-" prefix (or any section prefix)
    # Example: "Section No and Name 1-Karupparayan Kovil Street Ward No-9"
    m_street = re.search(r"Section\s+No\s+and\s+Name\s*[:\-]?\s*(.+)$", line2, re.I)
    if m_street:
        result["street"] = m_street.group(1).strip()

    return result


def parse_filename(filename: str) -> ParsedFile | None:
    m = FILENAME_RE.match(filename)
    if not m:
        return None

    # read the street .txt file from '../jpg' folder
    txt_filename = os.path.join("jpg", filename.replace("stacked_ocr", "street"))
    metadata = {}
    with open(txt_filename, encoding="utf-8") as f:
        metadata_text = f.read()
        lang = detect_ocr_language_from_filename(filename)
        if lang.startswith("tam"):
            metadata = parse_page_metadata_tamil(metadata_text)
        else:
            metadata = parse_page_metadata(metadata_text)
            logger.info(f"Parsed metadata for {filename}: {metadata}")

    return ParsedFile(
        doc_id=m.group("doc"),
        page_no=int(m.group("page")),
        assembly=metadata.get("assembly"),
        part_no=metadata.get("part_no"),
        street=metadata.get("street"),
    )


def _ocr_worker(crop, crop_name: str, lang: str) -> dict:
    """
    Worker function executed in a thread.
    """
    try:
        ocr_text = extract_text_from_image(crop, lang=lang)

        if not ocr_text.strip():
            # Return empty result if OCR text is empty
            return {
                "source_image": crop_name,
                "ocr_text": "",
                "epic_id": None,
            }

        epic_id = extract_epic_id(crop)

        return {
            "source_image": crop_name,
            "ocr_text": ocr_text,
            "epic_id": epic_id,
        }
    except Exception as e:
        logger.error(f"OCR failed for {crop_name}: {e}")
        return {
            "source_image": crop_name,
            "ocr_text": "",
            "epic_id": None,
        }


def extract_voters_from_stacked_txt_files(crops_dir: str, progress=None, limit=None) -> list[dict]:
    """
    Extracts voter information from stacked text files corresponding to cropped images.
    """

    files = sorted(f for f in os.listdir(crops_dir) if f.lower().endswith(".txt"))

    if limit:
        files = files[:limit]

    task = None
    if progress:
        task = progress.add_task("🔍 Crops -> OCR", total=len(files))

    results = []
    start_time = time.perf_counter()

    logger.info(f"🔍 Starting voter extraction from stacked text files ({len(files)} crops)")

    for file in files:
        path = os.path.join(crops_dir, file)

        with open(path, encoding="utf-8") as f:
            ocr_text = f.read()

        if not ocr_text.strip():
            if progress and task:
                progress.advance(task)
            continue

        parsed = parse_filename(file)
        if not parsed:
            logger.warning(f"⚠️ Failed to parse metadata from filename {file}, skipping.")
            if progress and task:
                progress.advance(task)
            continue

        results.append(
            {
                "source_image": file,
                "ocr_text": ocr_text,
                "doc_id": parsed.doc_id,
                "assembly": parsed.assembly,
                "part_no": parsed.part_no,
                "street": parsed.street,
                "page_no": parsed.page_no,
            }
        )

        if progress and task:
            progress.advance(task)

    with open("ocr/ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed_time = time.perf_counter() - start_time

    logger.info(f"Total number of voter results: {len(results)}")
    logger.info(f"Time taken by extract_voters_from_stacked_txt_files: {elapsed_time:.3f} seconds.")

    return results


def extract_ocr_from_crops_in_parallel(
    total_crops: list[dict], progress=None, max_workers=4, limit=None
):
    """
    Performs OCR on all cropped voter images using multi-threading.
    """

    # total_crops is a list of dicts with keys: crop_name, crop
    # sort total_crops by crop_name first
    sorted_crops = sorted(total_crops, key=lambda x: x["crop_name"])

    if limit:
        sorted_crops = sorted_crops[:limit]

    task = None
    if progress:
        task = progress.add_task("🔍 Crops -> OCR", total=len(sorted_crops))

    results = []
    start_time = time.perf_counter()

    logger.info(f"🔍 Starting OCR extraction ({len(sorted_crops)} crops, {max_workers} threads)")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_ocr_worker, crop["crop"], crop["crop_name"], crop["lang"]): crop[
                "crop_name"
            ]
            for crop in sorted_crops
        }

        for future in as_completed(futures):
            file = futures[future]
            result = future.result()

            if progress:
                progress.advance(task)

            if not result["ocr_text"].strip():
                continue

            parsed = parse_filename(result["source_image"])
            if not parsed:
                logger.warning(f"⚠️ Failed to parse metadata from filename {file}, skipping.")
                continue

            result["doc_id"] = parsed.doc_id
            result["assembly"] = parsed.assembly
            result["part_no"] = parsed.part_no
            result["street"] = parsed.street
            result["page_no"] = parsed.page_no
            result["voter_no"] = parsed.voter_no

            results.append(result)

    # Persist debug output
    with open("ocr/ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed_time = time.perf_counter() - start_time

    logger.info(f"Total number of OCR results: {len(results)}")
    logger.info(f"Time taken by extract_ocr_from_crops: {elapsed_time:.3f} seconds.")

    return results


def assign_serial_numbers(results: list[dict]) -> list[dict]:
    """
    Assigns serial_no resetting per doc_id.
    """

    # Group by doc_id
    grouped = defaultdict(list)
    for r in results:
        grouped[r["doc_id"]].append(r)

    if isDebugMode():
        logger.debug(f"Assigning serial numbers for {len(grouped)} documents")

    final = []

    for _, voters in grouped.items():
        # Sort deterministically inside doc
        voters.sort(key=lambda x: (x["page_no"]))

        # Assign serial_no
        for idx, voter in enumerate(voters, start=1):
            if isDebugMode():
                logger.debug(
                    f"Assigning serial_no {idx} to voter from doc {voter['doc_id']} (page {voter['page_no']})"
                )
            voter["serial_no"] = idx
            final.append(voter)

    # Optional: global stable ordering
    final.sort(key=lambda x: (x["doc_id"], x["serial_no"]))
    return final
