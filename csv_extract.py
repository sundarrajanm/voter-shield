import os
import time
import re
import json
from typing import List, Dict

def remove_unwanted_words(ocr_text, noise_words):
    cleaned = ocr_text
    for word in noise_words:
        cleaned = cleaned.replace(word, "")

    return cleaned

def remove_unwanted_lines_containing(ocr_text, noise_lines):
    lines = ocr_text.splitlines()
    cleaned_lines = [
        line for line in lines
        if not any(noise_line in line for noise_line in noise_lines)
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

FIELDS = [
    "Name",
    "Father Name",
    "Mother Name",
    "Husband Name",
    "House Number",
    "Age",
    "Gender"
]

def extract_values(line, field):
    pattern = (
        rf"{field}\s*[:=+]\s*"
        rf"(.*?)"
        rf"(?=\s+(?:{'|'.join(map(re.escape, FIELDS))})\s*[:=+]|$)"
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

# def parse_ocr_block(ocr_text: str):
#     voters = []
#     current = [ {}, {}, {} ]  # 3 voters per row

#     lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]

#     for line in lines:
#         # 🔑 CRITICAL FIX: only Name at START opens a new group
#         if line.startswith("Name"):
#             if any(v for v in current):
#                 voters.extend(current)
#                 current = [ {}, {}, {} ]

#             names = extract_values(line, "Name")
#             for i in range(min(3, len(names))):
#                 current[i]["name"] = names[i] or None

#             # 🔑 CAPTURE COLUMN SPANS HERE
#             column_spans = get_column_spans(line)                
#             continue  # do NOT fall through

#         if column_spans and any(x in line for x in ["Father Name", "Husband Name", "Mother Name"]):
#             for i, (start, end) in enumerate(column_spans):
#                 segment = line[start:end].strip()

#                 if "Father Name" in segment:
#                     val = extract_values(segment, "Father Name")
#                     current[i]["father_name"] = val[0] if val else None

#                 if "Husband Name" in segment:
#                     val = extract_values(segment, "Husband Name")
#                     current[i]["husband_name"] = val[0] if val else None

#                 if "Mother Name" in segment:
#                     val = extract_values(segment, "Mother Name")
#                     current[i]["mother_name"] = val[0] if val else None

#         if "House Number" in line:
#             vals = extract_values(line, "House Number")
#             for i in range(min(3, len(vals))):
#                 current[i]["house_no"] = vals[i] or None

#         if "Age" in line and "Gender" in line:
#             ages = re.findall(r"Age\s*[:+]\s*(\d+)", line)
#             genders = re.findall(r"Gender\s*[:+]\s*(Male|Female)", line, flags=re.IGNORECASE)

#             for i in range(min(3, len(ages), len(genders))):
#                 current[i]["age"] = int(ages[i])
#                 current[i]["gender"] = genders[i].capitalize()

#     # Flush last group
#     voters.extend(current)

#     # Normalize output
#     normalized = []
#     for v in voters:
#         normalized.append({
#             "name": v.get("name"),
#             "father_name": v.get("father_name"),
#             "mother_name": v.get("mother_name"),
#             "husband_name": v.get("husband_name"),
#             "house_no": v.get("house_no"),
#             "age": v.get("age"),
#             "gender": v.get("gender")
#         })

#     return [v for v in normalized if any(v.values())]

def parse_ocr_text(ocr_text: str) -> List[Dict]:
    voters = []

    # ✅ Robust row split (handles \n\n, \n \n, etc.)
    rows = re.split(r"\n\s*\n", ocr_text.strip())

    for row in rows:
        lines = [l.strip() for l in row.splitlines() if l.strip()]
        if len(lines) < 4:
            continue

        name_line, rel_line, house_line, age_line = lines[:4]

        # --- Names (do NOT depend on '-' existing) ---
        names = re.findall(r"Name\s*[:=]\s*([^:]+?)(?=\s+Name|$)", name_line)
        names = [n.strip() for n in names]

        # --- Relationships (preserve order) ---
        rels = re.findall(
            r"(Father Name|Mother Name|Husband Name|Other)\s*[:=]\s*([^:]+?)(?=\s+(Father|Mother|Husband|Other)\s+Name|$)",
            rel_line
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
                "gender": genders[i].capitalize() if i < len(genders) else None
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

# def clean_and_extract_csv(ocr_texts):
#     start_time = time.perf_counter()

#     # Remove "Photo" and "Available" words from OCR texts
#     noise_words = ["Photo", "Available"]
#     cleaned_records = [remove_unwanted_words(text, noise_words) for text in ocr_texts]

#     # Replace common OCR misreads with corrections
#     corrections = {
#         "Narne": "Name",
#         "Narme": "Name",
#         "Famale": "Female",
#         "Gander": "Gender"
#     }
#     cleaned_records = [replace_noise_words_with_corrections(text, corrections) for text in cleaned_records]    

#     # Remove unwanted lines
#     noise_lines_contains = [
#         "Assembly Constituency No and Name",
#         "Section No and Name",
#         "Date of Publication"
#     ]
#     cleaned_records = [remove_unwanted_lines_containing(text, noise_lines_contains) for text in cleaned_records]
        
#     # Write results to ocr_results_cleaned_before.json for inspection
#     with open("ocr/ocr_results_cleaned_before.json", "w", encoding="utf-8") as f:
#         json.dump(cleaned_records, f, ensure_ascii=False, indent=2)

#     # Parse each cleaned OCR block into structured voter records
#     # cleaned_records = [parse_ocr_block(text) for text in cleaned_records]
#     cleaned_records = [parse_ocr_text(text) for text in cleaned_records]
#     cleaned_records = [voter for block in cleaned_records for voter in block]  # flatten list

#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time
#     print(f"Time taken by clean_and_extract_csv: {elapsed_time:.6f} seconds.")

#     # Write results to ocr_results_cleaned.json for inspection
#     with open("ocr/ocr_results_cleaned.json", "w", encoding="utf-8") as f:
#         json.dump(cleaned_records, f, ensure_ascii=False, indent=2)

#     return cleaned_records

def clean_and_extract_csv(ocr_results):
    start_time = time.perf_counter()

    os.makedirs("ocr/cleaned_before", exist_ok=True)
    os.makedirs("ocr/parsed_after", exist_ok=True)

    all_voters = []

    noise_words = ["Photo", "Available"]

    corrections = {
        "Narne": "Name",
        "Narme": "Name",
        "Famale": "Female",
        "Gander": "Gender"
    }

    noise_lines_contains = [
        "Assembly Constituency No and Name",
        "Section No and Name",
        "Date of Publication"
    ]

    for item in ocr_results:
        source_image = item["source_image"]
        ocr_text = item["ocr_text"]

        # 1️⃣ Clean OCR text
        text = remove_unwanted_words(ocr_text, noise_words)
        text = replace_noise_words_with_corrections(text, corrections)
        text = remove_unwanted_lines_containing(text, noise_lines_contains)
        text = remove_epic_id_noise(text)

        # 2️⃣ Dump cleaned OCR (before parsing)
        cleaned_path = f"ocr/cleaned_before/{source_image}.json"
        with open(cleaned_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_image": source_image,
                    "cleaned_ocr_text": text
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        # 3️⃣ Parse voters from this page
        try:
            voters = parse_ocr_text(text)
        except Exception as e:
            print(f"❌ Parsing failed for {source_image}: {e}")
            voters = []

        # 4️⃣ Attach source_image to every voter
        for v in voters:
            v["source_image"] = source_image
            
            FIELDS = ["name", "father_name", "mother_name", "husband_name", "other_name"]
            if any(v.get(f) not in (None, "", " ") for f in FIELDS):
                all_voters.append(v)

        # 5️⃣ Dump parsed voters for this page
        parsed_path = f"ocr/parsed_after/{source_image}.json"
        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(voters, f, ensure_ascii=False, indent=2)

    end_time = time.perf_counter()
    print(f"⏱️ clean_and_extract_csv completed in {end_time - start_time:.3f} sec")
    print(f"📊 Total voters extracted: {len(all_voters)}")

    return all_voters
