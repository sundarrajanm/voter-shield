import json
import re
from typing import List, Dict

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

input = "Name : Rangaraj - Name : Lakshmi - Name : Kalaivani -\nFather Name: Makali - Husband Name: Rangaraj - Father Name: Rangaraj -\nHouse Number : 1 House Number : 1 House Number : 1 \nAge : 68 Gender : Male Age + 57 Gender : Female Age : 41 Gender : Female\n \nName : jothiswari - Name : ANANDHAKUMAR Name : GANESH\nHusband Name: Ramesh - Father Name: GNANAVEL Father Name: YANAVEL\nHouse Number : 1 House Number : 1 House Number : 1 \nAge : 37 Gender : Female Age : 31 Gender : Male Age : 27 Gender : Male\n"
results = parse_ocr_text(input)
print(json.dumps(results, indent=2))