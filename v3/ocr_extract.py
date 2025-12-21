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

import pytesseract
import re
from PIL import Image

def extract_epic_id_in_column(
    column_crop: Image.Image,
    rows: int = 10
) -> list[str]:
    """
    Extract EPIC IDs from a column crop.
    Returns a list of EPIC IDs in top-to-bottom order.
    """

    W, H = column_crop.size
    row_h = H / rows

    epic_ids = []

    for r in range(rows):
        # --- Crop single voter row ---
        top = int(r * row_h)
        bottom = int((r + 1) * row_h)
        row_crop = column_crop.crop((0, top, W, bottom))

        # --- EPIC region: right 40%, top ~35% of row ---
        rw, rh = row_crop.size
        x1 = int(rw * 0.60)
        y1 = 0
        x2 = rw
        y2 = int(rh * 0.35)

        epic_region = row_crop.crop((x1, y1, x2, y2))

        # --- OCR ---
        text = pytesseract.image_to_string(
            epic_region,
            lang="eng",
            config=(
                "--psm 7 --oem 1 "
                "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
        )

        # --- Clean ---
        text = "".join(c for c in text if c.isalnum()).strip()

        # --- Basic EPIC validation (Indian format heuristic) ---
        if re.match(r"^[A-Z]{3}\d{7}$", text):
            epic_ids.append(text)
        else:
            epic_ids.append("")  # preserve alignment

    return epic_ids

def extract_text_from_image(crop) -> str:
    text = pytesseract.image_to_string(crop, lang="eng", config="--psm 6 --oem 1")
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
from typing import List, NamedTuple

class ParsedFile(NamedTuple):
    doc_id: str
    page_no: int
    voter_no: int
    assembly: str
    part_no: int
    street: str

FILENAME_RE = re.compile(
    r"^(?P<doc>.+?)_page_(?P<page>\d+)_column_(?P<column>\d+)\.png$",
    re.IGNORECASE
)

import re
from typing import Dict, Optional

def parse_page_metadata(ocr_text: str) -> Dict[str, Optional[str]]:
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

    # read a .txt file from '../png' folder and replace '_voter_xx' with empty in txt_filename
    txt_filename = os.path.join("png", re.sub(r"_voter_\d+", "", filename).replace(".png", "_street.txt"))
    metadata = {}
    with open(txt_filename, "r", encoding="utf-8") as f:
        metadata_text = f.read()
        metadata = parse_page_metadata(metadata_text)
    
    return ParsedFile(
        doc_id=m.group("doc"),
        page_no=int(m.group("page")),
        voter_no=int(m.group("voter")),
        assembly=metadata.get("assembly"),
        part_no=metadata.get("part_no"),
        street=metadata.get("street")
    )

def parse_column_voter_ocr_v2(column_ocr_text: str, serial_no_start, street, part_no, assembly, age_as_on, published_on, crop_name, ocr_text) -> List[Dict[str, Optional[str]]]:
    """
    Parse OCR text of a SINGLE COLUMN (up to 10 voters) into structured voter records.
    """

    if not column_ocr_text or "Name" not in column_ocr_text:
        return []

    # ------------------------------------------------------------------
    # 1️⃣ Normalize common OCR mistakes
    # ------------------------------------------------------------------
    corrections = {
        "Narne": "Name",
        "Narme": "Name",
        "Famale": "Female",
        "Gander": "Gender",
    }
    for wrong, correct in corrections.items():
        column_ocr_text = column_ocr_text.replace(wrong, correct)

    # ------------------------------------------------------------------
    # 2️⃣ Remove noise BEFORE first Name (EPIC IDs, garbage, etc.)
    # ------------------------------------------------------------------
    first_name_idx = column_ocr_text.find("Name")
    if first_name_idx == -1:
        return []

    column_ocr_text = column_ocr_text[first_name_idx:]

    # ------------------------------------------------------------------
    # 3️⃣ Tokenize using KEY-AWARE regex (this is the 핵심)
    # ------------------------------------------------------------------
    KEY_PATTERN = (
        r"(Name|Father Name|Mother Name|Husband Name|Other|House Number|Age|Gender)"
        r"\s*[:=+]\s*"
    )

    tokens = re.split(KEY_PATTERN, column_ocr_text)

    # tokens example:
    # ["", "Name", "Sahar Banu -\n", "Husband Name", "Abdul Wahab -\n", ...]

    # ------------------------------------------------------------------
    # 4️⃣ Convert tokens → ordered (key, value) pairs
    # ------------------------------------------------------------------
    pairs = []
    token_iter = iter(tokens[1:])  # skip junk before first key

    for key, value in zip(token_iter, token_iter):
        clean_value = value.strip()
        pairs.append((key.strip(), clean_value))

    # ------------------------------------------------------------------
    # 5️⃣ Build voters using a state machine
    # ------------------------------------------------------------------
    voters = []
    current = None

    for key, value in pairs:
        if key == "Name":
            # Flush previous voter
            if current:
                voters.append(current)

            current = {
                "serial_no": serial_no_start,
                "epic_id": None,
                "name": value,
                "father_name": None,
                "mother_name": None,
                "husband_name": None,
                "other_name": None,
                "house_no": None,
                "age": None,
                "gender": None,

                # street, part_no, assembly, age_as_on, published_on
                "street": street,
                "part_no": part_no,
                "assembly": assembly,
                "age_as_on": age_as_on,
                "published_on": published_on,
                "source_image": crop_name,
                "ocr_text": ocr_text,
            }

            serial_no_start += 3

        elif current:
            if key == "Father Name":
                current["father_name"] = value

            elif key == "Mother Name":
                current["mother_name"] = value

            elif key == "Husband Name":
                current["husband_name"] = value

            elif key == "Other":
                current["other_name"] = value

            elif key == "House Number":
                current["house_no"] = value

            elif key == "Age":
                m = re.search(r"\d+", value)
                if m:
                    current["age"] = int(m.group())

            elif key == "Gender":
                # Store as M / F
                current["gender"] = value[0].upper()

    # Flush last voter
    if current:
        voters.append(current)

    # ------------------------------------------------------------------
    # 6️⃣ Final cleanup (drop empty voters defensively)
    # ------------------------------------------------------------------
    cleaned = []
    for v in voters:
        if any(v.get(k) for k in ["name", "father_name", "husband_name", "mother_name", "other_name"]):
            cleaned.append(v)

    return cleaned

import re
from typing import Optional, Dict

def extract_assembly_and_part_no(source_image: str) -> Dict[str, Optional[int]]:
    """
    Extracts assembly and part number from voter image filename.

    Works for dynamic districts:
    Example:
    2025-EROLLGEN-S22-115-FinalRoll-Revision1-ENG-186-WI_page_04_column_1.png
    """

    assembly = None
    part_no = None

    # S<district>-<assembly> ... ENG-<part>
    m = re.search(r"S\d+-(\d+).*?ENG-(\d+)", source_image)
    if m:
        assembly = int(m.group(1))
        part_no = int(m.group(2))

    return {
        "assembly": assembly,
        "part_no": part_no
    }

def _ocr_worker(crop_dict) -> dict:
    """
    Worker function executed in a thread.
    """
    try:
        ocr_text = extract_text_from_image(crop_dict["crop"])
        # epic_id = extract_epic_id(crop)
        epic_ids = extract_epic_id_in_column(crop_dict["crop"])

        metadata = extract_assembly_and_part_no(crop_dict["crop_name"])
        voters = parse_column_voter_ocr_v2(ocr_text,
                                           serial_no_start=1, 
                                           part_no=metadata["part_no"],
                                           assembly=metadata["assembly"], 
                                           street=crop_dict.get("page_metadata", {}).get("street"), 
                                           age_as_on=crop_dict.get("page_metadata", {}).get("age_as_on"),
                                           published_on=crop_dict.get("page_metadata", {}).get("published_on"),
                                           crop_name=crop_dict["crop_name"],
                                           ocr_text=ocr_text
                                           )
        
        for i, voter in enumerate(voters):
            voter["epic_id"] = epic_ids[i] if i < len(epic_ids) else ""

        return {
            "ocr_text": ocr_text,
            "voters": voters,
        }

    except Exception as e:
        logger.error(f"OCR failed for {crop_name}: {e}")
        return {
            "source_image": crop_name,
            "ocr_text": "",
            "epic_id": None,
        }

from typing import List, Dict

from typing import List, Dict

def assign_serial_no_per_part(records: List[Dict]) -> None:
    """
    Sort by (part_no, source_image) and assign serial_no.
    serial_no resets to 1 whenever part_no changes.
    Mutates records in place.
    """

    # 1️⃣ Sort by part_no first, then source_image
    records.sort(key=lambda r: (r.get("part_no"), r.get("source_image", "")))

    current_part = None
    serial = 0

    # 2️⃣ Assign serial_no with reset on part_no change
    for record in records:
        part_no = record.get("part_no")

        if part_no != current_part:
            current_part = part_no
            serial = 1
        else:
            serial += 1

        record["serial_no"] = serial

def extract_ocr_from_crops_in_parallel(total_crops: list[dict], progress=None, max_workers=4, limit=None):
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
        task = progress.add_task(f"🔍 Crops -> OCR", total=len(sorted_crops))

    results = []
    start_time = time.perf_counter()

    logger.info(
        f"🔍 Starting OCR extraction ({len(sorted_crops)} crops, {max_workers} threads)"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_ocr_worker, crop): crop["crop_name"]
            for crop in sorted_crops
        }

        for future in as_completed(futures):
            result = future.result()

            if progress:
                progress.advance(task)

            if not result["ocr_text"].strip():
                continue

            results.extend(result["voters"])

    assign_serial_no_per_part(results)

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

    for _, voters in grouped.items():
        # Sort deterministically inside doc
        voters.sort(key=lambda x: (x["page_no"], x["voter_no"]))

        # Assign serial_no
        for idx, voter in enumerate(voters, start=1):
            voter["serial_no"] = idx
            final.append(voter)

    # Optional: global stable ordering
    final.sort(key=lambda x: (x["doc_id"], x["serial_no"]))
    return final
