import time
from PIL import Image, ImageDraw
import os
import pytesseract
from PIL import Image
import json, requests
import csv

SYSTEM_PROMPT = """
You are a strict JSON data extractor.
Follow these rules:

1. Always return valid JSON only.
2. Do not guess any missing information; return null for missing fields.
3. Remove noise characters like '-', '+', '~', '=', '*', ':'.
4. Map:
   - "Father Name" → "father_name"
   - "Mother Name" → "mother_name"
   - "Husband Name" → "husband_name"
5. The voter name (after "Name") goes into "name".
6. Extract house number, age, and gender.
7. Normalize gender to "Male" or "Female".
8. Never hallucinate values.
"""

def clean_with_llm(ocr_text, epic_id):
    user_prompt = f"""
        Extract structured voter data.

        OCR:
        \"\"\"
        {ocr_text}
        \"\"\"

        EPIC_ID: {epic_id}

        Return JSON with this structure:
        {{
        "epic_id": "",
        "name": "",
        "father_name": "",
        "mother_name": "",
        "husband_name": "",
        "house_no": "",
        "age": "",
        "gender": ""
        }}
        """

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json = {
            "model": "qwen2.5:7b-instruct",
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt,
            "format": "json",
            "stream": False
        }
    ).json()

    return json.loads(resp["response"])

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def extract_text_from_image(image_path: str) -> str:
    print(f"\n🔍 OCR processing: {image_path}")
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng", config="--psm 6 --oem 1")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    print(f"📄 OCR extracted {len(lines)} non-empty lines.")
    return "\n".join(lines)

def extract_epic_id(crop, voter_id=None):
    cw, ch = crop.size

    # Crop the entire right 40%
    x1 = int(cw * 0.60)   # left boundary for EPIC region
    x2 = cw               # till the rightmost edge
    y1 = 0
    y2 = ch

    epic_region = crop.crop((x1, y1, x2, y2))

    # Optional: store debug image
    if voter_id is not None:
        debug_dir = "debug_epic_right40"
        os.makedirs(debug_dir, exist_ok=True)
        epic_region.save(f"{debug_dir}/voter_{voter_id:02d}_epic.png")

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

def append_to_csv(data: dict):
    # Create the CSV file with header if it doesn't exist
    file_exists = os.path.isfile(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)

        if not file_exists:
            writer.writeheader()

        # Convert None → "" for CSV cleanliness
        sanitized = {k: ("" if v is None else v) for k, v in data.items()}

        writer.writerow(sanitized)

def crop_voter_boxes_dynamic(input_png, out_dir="crops"):
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(input_png)

    W, H = img.size
    print(f"Processing image {input_png} ({W} x {H})")

    # DPI-invariant margins
    top_header_pct = 0.032
    bottom_footer_pct = 0.032
    left_margin_pct = 0.024
    right_margin_pct = 0.024

    top_header = int(H * top_header_pct)
    bottom_footer = int(H * bottom_footer_pct)
    left_margin = int(W * left_margin_pct)
    right_margin = int(W * right_margin_pct)

    content_x = left_margin
    content_y = top_header
    content_w = W - left_margin - right_margin
    content_h = H - top_header - bottom_footer

    ROWS, COLS = 10, 3
    box_w = content_w / COLS
    box_h = content_h / ROWS

    # Photo box proportional values
    photo_x_ratio = 380 / 1555
    photo_w_ratio = 380 / 1555
    photo_h_ratio = 480 / 620
    photo_y_ratio = (620 - 480) / 620  # starting point

    count = 1

    for r in range(ROWS):
        for c in range(COLS):
            # Record the start time
            start_time = time.perf_counter() # Or time.time() for less precision

            left = int(content_x + c * box_w)
            upper = int(content_y + r * box_h)
            right = int(left + box_w)
            lower = int(upper + box_h)

            crop = img.crop((left, upper, right, lower))

            cw, ch = crop.size

            # Convert ratios to actual DPI-scaled pixels
            px_left = int(cw * (1 - photo_w_ratio))
            px_top = int(ch * photo_y_ratio)
            px_right = cw
            px_bottom = int(ch)

            # Add padding to fully remove photo-box borders
            pad_x = int(cw * 0.02)
            pad_y = int(ch * 0.02)

            px_left   = max(0, px_left - pad_x)
            px_top    = max(0, px_top - pad_y)
            px_right  = min(cw, px_right + pad_x)
            px_bottom = min(ch, px_bottom + pad_y)

            draw = ImageDraw.Draw(crop)
            draw.rectangle([px_left, px_top, px_right, px_bottom], fill="white")

            crop.save(f"{out_dir}/voter_{count:02d}.png")
            print(f"Saved voter_{count:02d}.png")

            ocr_text = extract_text_from_image(f"{out_dir}/voter_{count:02d}.png")
            epic_id = extract_epic_id(crop, count)
            cleaned_data = clean_with_llm(ocr_text, epic_id)
            append_to_csv(cleaned_data)
            print(f"Cleaned Data for voter_{count:02d}:\n{json.dumps(cleaned_data, indent=2)}")

            # Record the end time
            end_time = time.perf_counter() # Or time.time()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time

            # Print the elapsed time
            print(f"Time taken for an voter: {elapsed_time:.3f} seconds.")

            count += 1

    print(f"Done! Total crops: {count - 1}")

# RUN
INPUT_PNG = "./png/page_03.png"
OUTPUT_CSV = "./page-csv/page_03.csv"
CSV_HEADERS = [
    "epic_id",
    "name",
    "father_name",
    "mother_name",
    "husband_name",
    "house_no",
    "age",
    "gender"
]


crop_voter_boxes_dynamic(INPUT_PNG)
