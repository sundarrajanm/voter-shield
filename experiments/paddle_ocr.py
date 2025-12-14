from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import os

# Initialize OCR (English only, no angle classification needed)
ocr = PaddleOCR(lang="en", use_textline_orientation=False)

import cv2
import numpy as np

def ensure_3_channel(img):
    """
    PaddleOCR requires 3-channel images (H, W, 3)
    """
    if len(img.shape) == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def remove_box_lines(img):
    """
    Removes horizontal & vertical box lines from voter images
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Binary inversion (important!)
    _, bw = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)
    bw = cv2.subtract(bw, vertical_lines)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel)
    bw = cv2.subtract(bw, horizontal_lines)

    # Invert back
    cleaned = cv2.bitwise_not(bw)
    return cleaned

def extract_serial_and_epic_paddle(image_path, debug_dir="debug_paddle"):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # 🔹 Crop top 20%
    top_h = int(h * 0.20)
    top_region = img.crop((0, 0, w, top_h))

    # 🔹 Split horizontally
    epic_region   = top_region.crop((int(w * 0.35), 0, w, top_h))

    os.makedirs(debug_dir, exist_ok=True)
    epic_region.save(f"{debug_dir}/epic_region.png")

    # Convert to numpy for PaddleOCR
    epic_np = np.array(epic_region)
    # epic_np = ensure_3_channel(epic_np)
    # epic_np = remove_box_lines(epic_np)

    epic_ocr   = ocr.predict(epic_np)

    def extract_text(ocr_result):
        if not ocr_result or not ocr_result[0]:
            return ""
        return " ".join([line[1][0] for line in ocr_result[0]])

    # serial_text = extract_text(serial_ocr)
    epic_text   = extract_text(epic_ocr)

    return {
        "serial_raw": "",
        "epic_raw": epic_text
    }


if __name__ == "__main__":
    image_path = "crops/2025-EROLLGEN-S22-116-FinalRoll-Revision1-ENG-244-WI_page_03_voter_01.png"
    result = extract_serial_and_epic_paddle(image_path)

    print("🔢 Serial OCR:", result["serial_raw"])
    print("🆔 EPIC OCR  :", result["epic_raw"])
