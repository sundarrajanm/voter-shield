import re
import time

from crop_voters import detect_ocr_language_from_filename
from logger import setup_logger
from utilities import parse_single_voter_ocr_tamil, split_voters_from_page_ocr

logger = setup_logger()


def remove_unwanted_words(ocr_text, noise_words):
    cleaned = ocr_text
    for word in noise_words:
        cleaned = cleaned.replace(word, "")

    return cleaned


def remove_unwanted_lines_containing(ocr_text, noise_lines):
    lines = ocr_text.splitlines()
    cleaned_lines = [
        line for line in lines if not any(noise_line in line for noise_line in noise_lines)
    ]
    return "\n".join(cleaned_lines)


def replace_noise_words_with_corrections(ocr_text, corrections):
    cleaned = ocr_text
    for wrong, right in corrections.items():
        cleaned = cleaned.replace(wrong, right)
    return cleaned


def remove_epic_id_noise(ocr_text: str) -> str:
    """
    Removes EPIC-ID noise lines that appear before Name blocks.
    Ensures every voter row starts with Name.
    """
    lines = ocr_text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        # Skip pure EPIC-ID noise lines
        # Example: NHH1675131 NHH3179512 NHH2225969
        if re.fullmatch(r"(?:[A-Z]{3}\d{7}\s*)+", line):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


FIELDS = ["Name", "Father Name", "Mother Name", "Husband Name", "House Number", "Age", "Gender"]


def extract_values(line, field):
    pattern = (
        rf"{field}\s*[:=+]\s*" rf"(.*?)" rf"(?=\s+(?:{'|'.join(map(re.escape, FIELDS))})\s*[:=+]|$)"
    )
    return [v.strip(" -+=~") for v in re.findall(pattern, line, flags=re.IGNORECASE)]


def get_column_spans(name_line):
    """
    Returns [(start, end), ...] for each voter column
    based on Name line positions.
    """
    starts = [m.start() for m in re.finditer(r"Name\s*:", name_line)]
    spans = []

    for i in range(len(starts)):
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(starts) else None
        spans.append((start, end))

    return spans


def parse_ocr_text(ocr_text: str) -> list[dict]:
    voters = []

    # ✅ Robust row split (handles \n\n, \n \n, etc.)
    rows = re.split(r"\n\s*\n", ocr_text.strip())

    for row in rows:
        lines = [line.strip() for line in row.splitlines() if line.strip()]
        if len(lines) < 4:
            continue

        name_line, rel_line, house_line, age_line = lines[:4]

        # --- Names (do NOT depend on '-' existing) ---
        names = re.findall(r"Name\s*[:=]\s*([^:]+?)(?=\s+Name|$)", name_line)
        names = [n.strip() for n in names]

        # --- Relationships (preserve order) ---
        rels = re.findall(
            r"(Father Name|Mother Name|Husband Name|Other)\s*[:=]\s*([^:]+?)(?=\s+(Father|Mother|Husband|Other)\s+Name|$)",
            rel_line,
        )

        # --- House Numbers ---
        houses = re.findall(r"House Number\s*[:=]\s*([A-Za-z0-9/]+)", house_line)

        # --- Age & Gender ---
        ages = re.findall(r"Age\s*[:+]\s*(\d+)", age_line)
        genders = re.findall(r"Gender\s*[:+]\s*(Male|Female)", age_line, re.I)

        # Build 3 voters per row
        for i in range(3):
            voter = {
                "name": names[i] if i < len(names) else None,
                "father_name": None,
                "mother_name": None,
                "husband_name": None,
                "house_no": houses[i] if i < len(houses) else None,
                "age": int(ages[i]) if i < len(ages) else None,
                "gender": genders[i].capitalize() if i < len(genders) else None,
            }

            # Assign relationship by column index
            if i < len(rels):
                label, value, _ = rels[i]
                value = value.strip()

                if label == "Father Name":
                    voter["father_name"] = value
                elif label == "Mother Name":
                    voter["mother_name"] = value
                elif label == "Husband Name":
                    voter["husband_name"] = value
                elif label == "Other":
                    voter["other_name"] = value

            voters.append(voter)

    return voters


def looks_like_epic_line(line: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", line)

    # Relaxed rule:
    # - at least 8 chars
    # - alphanumeric
    # - starts with letters (most EPICs do)
    return bool(re.match(r"[A-Z]{2,}\d{5,}", cleaned, re.IGNORECASE))


# Given OCR text for a single page, parse and return list of voter dicts
def parse_per_page_ocr_text(ocr_text: str, lang, limit=None) -> list[dict]:
    voter_texts = split_voters_from_page_ocr(ocr_text)
    voters = []

    for vt in voter_texts:
        if lang == "tam+eng":
            voter = parse_single_voter_ocr_tamil(vt)
        else:
            voter = parse_single_voter_ocr(vt)

        # voter["ocr_block"] = vt  # for debugging

        # if not voter["epic_id"]:
        #     logger.info(f"⚠️ EPIC ID missing for voter OCR:\n{vt}\n---")

        voters.append(voter)
        if limit is not None and len(voters) >= limit:
            break

    return voters


def clean_and_extract_csv_v2(ocr_results, progress=None, limit=None):
    start_time = time.perf_counter()

    all_voters = []
    task = None

    # Limit for testing
    if limit is not None:
        ocr_results = ocr_results[:limit]

    if progress:
        task = progress.add_task("🧠 OCR -> CSV", total=len(ocr_results))

    for item in ocr_results:
        if progress:
            progress.advance(task)

        lang = detect_ocr_language_from_filename(item["source_image"])
        per_page_voters = parse_per_page_ocr_text(item["ocr_text"], lang=lang)
        for v in per_page_voters:
            v.update(
                {
                    "source_image": item["source_image"],
                    "doc_id": item.get("doc_id"),
                    "assembly": item.get("assembly"),
                    "part_no": item.get("part_no"),
                    "street": item.get("street"),
                    "page_no": item.get("page_no"),
                }
            )

            all_voters.append(v)

    end_time = time.perf_counter()

    logger.info(f"⏱️ clean_and_extract_csv_v2 completed in {end_time - start_time:.3f} sec")
    logger.info(f"📊 Total voters extracted: {len(all_voters)}")

    return all_voters


def clean_and_extract_csv(ocr_results, progress=None):
    start_time = time.perf_counter()

    all_voters = []
    task = None
    if progress:
        task = progress.add_task("🧠 OCR -> CSV", total=len(ocr_results))
    for item in ocr_results:
        if progress:
            progress.advance(task)

        voter = parse_single_voter_ocr(item["ocr_text"])
        # voter = parse_single_voter_ocr_multilang(item["ocr_text"])

        voter.update(item)
        all_voters.append(voter)

    end_time = time.perf_counter()

    logger.info(f"⏱️ clean_and_extract_csv completed in {end_time - start_time:.3f} sec")
    logger.info(f"📊 Total voters extracted: {len(all_voters)}")

    return all_voters


def parse_single_voter_ocr(ocr_text: str) -> dict[str, str | None]:
    """
    Parse OCR text of a SINGLE voter box into structured fields,
    including EPIC ID.
    """

    result = {
        "epic_id": None,
        "name": None,
        "father_name": None,
        "husband_name": None,
        "mother_name": None,
        "other_name": None,
        "house_no": None,
        "age": None,
        "gender": None,
    }

    corrections = {
        "Narne": "Name",
        "Narme": "Name",
        "Narnme": "Name",
        "Famale": "Female",
        "Gander": "Gender",
    }
    ocr_text = replace_noise_words_with_corrections(ocr_text, corrections)

    if not ocr_text or "Name" not in ocr_text:
        return result

    # Remove noise BEFORE first 'Name'
    ocr_text = ocr_text[ocr_text.find("Name") :]

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    epic_candidates = []

    for line in lines:
        # --- NAME ---
        if line.startswith("Name"):
            m = re.search(r"Name\s*[:=]\s*(.+)", line)
            if m:
                result["name"] = m.group(1).strip()

        # --- FATHER ---
        elif line.startswith("Father"):
            m = re.search(r"Father\s+Name\s*[:=]\s*(.+)", line)
            if m:
                result["father_name"] = m.group(1).strip()

        # --- MOTHER ---
        elif line.startswith("Mother"):
            m = re.search(r"Mother\s+Name\s*[:=]\s*(.+)", line)
            if m:
                result["mother_name"] = m.group(1).strip()

        # --- HUSBAND ---
        elif line.startswith("Husband"):
            m = re.search(r"Husband\s+Name\s*[:=]\s*(.+)", line)
            if m:
                result["husband_name"] = m.group(1).strip()

        # --- OTHER ---
        elif line.startswith("Other"):
            m = re.search(r"Other\s*[:=]\s*(.+)", line)
            if m:
                result["other_name"] = m.group(1).strip()

        # --- HOUSE NUMBER ---
        elif "House Number" in line:
            m = re.search(r"House\s+Number\s*[:=]\s*([A-Za-z0-9/().,-]+)", line)
            if m:
                result["house_no"] = m.group(1).strip()

        # --- AGE + GENDER ---
        elif "Age" in line and "Gender" in line:
            age_m = re.search(r"Age\s*[:+]\s*(\d+)", line)
            gender_m = re.search(r"Gender\s*[:+]\s*(Male|Female)", line, re.I)

            if age_m:
                result["age"] = int(age_m.group(1))
            if gender_m:
                result["gender"] = gender_m.group(1)[0].upper()

        # --- EPIC ID CANDIDATE ---
        else:
            # Remove junk characters
            cleaned = re.sub(r"[^A-Za-z0-9]", "", line)

            # EPIC IDs are usually uppercase alphanumeric, length ~10
            if (
                cleaned
                and cleaned.upper() == cleaned
                and any(c.isdigit() for c in cleaned)
                and len(cleaned) >= 8
            ):
                epic_candidates.append(cleaned)

    # Pick the most likely EPIC (longest alphanumeric token)
    if epic_candidates:
        result["epic_id"] = max(epic_candidates, key=len)

    return result


def normalize_gender(value: str) -> str | None:
    v = value.strip().lower()

    if any(k in v for k in ["male", "m", "ஆண்"]):
        return "M"
    if any(k in v for k in ["female", "f", "பெண்"]):
        return "F"

    return None


# def parse_single_voter_ocr_multilang(ocr_text: str) -> dict[str, str | None]:
#     result = {
#         "name": None,
#         "father_name": None,
#         "husband_name": None,
#         "mother_name": None,
#         "other_name": None,
#         "house_no": None,
#         "age": None,
#         "gender": None,
#     }

#     if not ocr_text:
#         return result

#     # 🔑 CRITICAL FIX
#     ocr_text = normalize_ocr_text(ocr_text)

#     lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

#     for line in lines:
#         # --- FIELD LABELS ---
#         for field, patterns in FIELD_LABELS.items():
#             for p in patterns:
#                 m = re.search(rf"{p}\s*[:=]\s*(.+)", line, re.IGNORECASE)
#                 if m:
#                     result[field] = m.group(1).strip()
#                     break

#         # --- AGE ---
#         age_m = re.search(r"(Age|வயது)\s*[:+]\s*(\d+)", line, re.IGNORECASE)
#         if age_m:
#             result["age"] = int(age_m.group(2))

#         # --- GENDER ---
#         gender_m = re.search(
#             r"(Gender|பாலினம்)\s*[:+]\s*([A-Za-z\u0B80-\u0BFF]+)",
#             line,
#             re.IGNORECASE,
#         )
#         if gender_m:
#             result["gender"] = normalize_gender(gender_m.group(2))

#     return result
