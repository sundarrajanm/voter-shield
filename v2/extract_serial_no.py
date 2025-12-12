import os
import cv2
import numpy as np
import pytesseract
from PIL import Image

def extract_serial_no(crop, voter_id=None):
    """
    Extract serial number by removing box lines before OCR.
    Region:
      - Left 40% width
      - Top 20% height
    """

    cw, ch = crop.size

    # Crop serial number region
    x1 = 0
    x2 = int(cw * 0.40)
    y1 = 0
    y2 = int(ch * 0.22)

    serial_region = crop.crop((x1, y1, x2, y2))

    # Convert PIL → OpenCV
    img = np.array(serial_region)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Binary inverse (digits become white)
    _, bw = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --- Remove vertical lines ---
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, bw.shape[0] // 2)
    )
    vertical_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)

    bw_no_vertical = cv2.subtract(bw, vertical_lines)

    # --- Remove horizontal lines ---
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (bw.shape[1] // 2, 1)
    )
    horizontal_lines = cv2.morphologyEx(bw_no_vertical, cv2.MORPH_OPEN, horizontal_kernel)

    cleaned = cv2.subtract(bw_no_vertical, horizontal_lines)

    # Optional debug outputs
    if voter_id is not None:
        debug_dir = "debug_serial_region"
        os.makedirs(debug_dir, exist_ok=True)

        cv2.imwrite(f"{debug_dir}/voter_{voter_id:02d}_01_gray.png", gray)
        cv2.imwrite(f"{debug_dir}/voter_{voter_id:02d}_02_bw.png", bw)
        cv2.imwrite(f"{debug_dir}/voter_{voter_id:02d}_03_cleaned.png", cleaned)

    # OCR (single line, digits only)
    serial_text = pytesseract.image_to_string(
        cleaned,
        lang="eng",
        config="--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ).strip()

    # # Final cleanup
    # serial_text = "".join(c for c in serial_text if c.isdigit())

    return serial_text


# Path to crops folder
CROPS_DIR = "crops"

for file in sorted(os.listdir(CROPS_DIR)):
    if file.lower().endswith(".png"):
        file_path = os.path.join(CROPS_DIR, file)

        try:
            crop = Image.open(file_path)

            # Try to extract voter index from filename like voter_01.png
            voter_id = None
            if "voter_" in file:
                voter_id = int(file.replace("voter_", "").replace(".png", ""))

            serial_no = extract_serial_no(crop, voter_id=voter_id)

            print(f"{file} → Serial No: {serial_no}")

        except Exception as e:
            print(f"{file} → ERROR: {e}")
