import json
import re
from typing import List, Dict, Optional

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
            r"(Father Name|Mother Name|Husband Name)\s*[:=]\s*([^:]+?)(?=\s+(Father|Mother|Husband)\s+Name|$)",
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

            voters.append(voter)

    return voters

def parse_column_voter_ocr(column_ocr_text: str, limit=None) -> List[Dict]:
    """
    Parses OCR text of a SINGLE COLUMN (up to 10 voters)
    and returns a list of structured voter dicts.
    """

    voters = []

    if not column_ocr_text or "Name" not in column_ocr_text:
        return voters

    # 1️⃣ Remove noise BEFORE first Name
    column_ocr_text = column_ocr_text[column_ocr_text.find("Name"):]

    # 2️⃣ Split into voter blocks using lookahead on "Name"
    voter_blocks = re.split(r"(?=\n?Name\s*[:=])", column_ocr_text)

    # parse only up to `limit` voters if specified
    if limit is not None:
        voter_blocks = voter_blocks[:limit]

    for block in voter_blocks:
        print("--- Voter Block ---")
        print(block)
        print("-------------------\n")

        block = block.strip()
        if not block:
            continue

        voter = parse_single_voter_ocr(block)

        # Keep only meaningful voters
        if any(voter.values()):
            voters.append(voter)

    return voters

def replace_noise_words_with_corrections(ocr_text, corrections):
    cleaned = ocr_text
    for wrong, right in corrections.items():
        cleaned = cleaned.replace(wrong, right)
    return cleaned

def parse_single_voter_ocr(ocr_text: str) -> Dict[str, Optional[str]]:
    """
    Parse OCR text of a SINGLE voter box into structured fields.
    """

    result = {
        "name": None,
        "father_name": None,
        "husband_name": None,
        "mother_name": None,
        "other_name": None,
        "house_no": None,
        "age": None,
        "gender": None,
    }

    # Replace common OCR misreads with corrections
    corrections = {
        "Narne": "Name",
        "Narme": "Name",
        "Famale": "Female",
        "Gander": "Gender"
    }
    ocr_text = replace_noise_words_with_corrections(ocr_text, corrections)

    if not ocr_text or "Name" not in ocr_text:
        return result

    # 1️⃣ Remove noise BEFORE first 'Name'
    ocr_text = ocr_text[ocr_text.find("Name"):]

    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]

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
            m = re.search(r"House\s+Number\s*[:=]\s*([A-Za-z0-9/]+)", line)
            if m:
                result["house_no"] = m.group(1)

        # --- AGE + GENDER ---
        elif "Age" in line and "Gender" in line:
            age_m = re.search(r"Age\s*[:+]\s*(\d+)", line)
            gender_m = re.search(r"Gender\s*[:+]\s*(Male|Female)", line, re.I)

            if age_m:
                result["age"] = int(age_m.group(1))
            if gender_m:
                result["gender"] = gender_m.group(1)[0].upper()  # M / F

    return result

import re
from typing import List, Dict, Optional


def parse_column_voter_ocr_v2(column_ocr_text: str, serial_no_start, street, part_no, assembly, age_as_on, published_on) -> List[Dict[str, Optional[str]]]:
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
                "name": value,
                "father_name": None,
                "mother_name": None,
                "husband_name": None,
                "other_name": None,
                "house_no": None,
                "age": None,
                "gender": None,

                # street, part_no, assembly, age_as_on, published_on
                "steet": street,
                "part_no": part_no,
                "assembly": assembly,
                "age_as_on": age_as_on,
                "published_on": published_on,
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

# input = "Name : Rangaraj - Name : Lakshmi - Name : Kalaivani -\nFather Name: Makali - Husband Name: Rangaraj - Father Name: Rangaraj -\nHouse Number : 1 House Number : 1 House Number : 1 \nAge : 68 Gender : Male Age + 57 Gender : Female Age : 41 Gender : Female\n \nName : jothiswari - Name : ANANDHAKUMAR Name : GANESH\nHusband Name: Ramesh - Father Name: GNANAVEL Father Name: YANAVEL\nHouse Number : 1 House Number : 1 House Number : 1 \nAge : 37 Gender : Female Age : 31 Gender : Male Age : 27 Gender : Male\n"
input = "[Fy WRK 1923283\nName : Sahar Banu -\nHusband Name: Abdul Wahab -\nHouse Number : 1\nAge + 73 Gender : Female\n[4] WRK1923275 |\nName : Mohamed Shafiullah -\nFather Name: Abdul Wahab -\nHouse Number : 1\nAge : 57 Gender : Male\nwrK3390499\nName : selva pandi\nFather Name: athikari\nHouse Number : 1\nAge : 47 Gender : Male\nName : Shibindas -\nFather Name: Kannadasan -\nHouse Number : 1\nAge : 37 Gender : Male\nName + Chitra -\nHusband Name: Jeyaprakash -\nHouse Number : 1-1244\nAge : 41 Gender : Female\nWRK0090464\nName : Umamaheswari -\nHusband Name: Duraisamy -\nHouse Number : 1/159 (43), SITE.NO-9\nAge : 53 Gender : Female\nName : Dhivyaa Lakshmi\nHusband Name: Ramesh\nHouse Number : 1/443 c 10\nAge : 43 Gender : Female\nName : Sakthivel\nFather Name: Sakthivel\nHouse Number : 1/6734\nAge : 49 Gender : Male\n[8] WRKO0725374\nName + Sukuna -\nHusband Name: Raja -\nHouse Number : 1E\nAge : 56 Gender : Female\nName : Ranipriya ~\nHusband Name: Gopalakrishnan -\nHouse Number : 1E\nAge : 50 Gender : Female"

# Add to output: street, part_no, assembly, age_as_on, published_on
results = parse_column_voter_ocr_v2(input, serial_no_start=1, street="Sample Street", part_no=1, assembly="Sample Assembly", age_as_on="01-01-2025", published_on="06-01-2025")
print(json.dumps(results, indent=2))
