import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytesseract
from PIL import Image, ImageDraw

from config import CROPS_DIR, VOTER_END_MARKER
from logger import setup_logger
from ocr_extract import extract_text_from_image

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


def extract_epic_region(crop, epic_x_ratio=0.60, epic_y_ratio=0.25):
    cw, ch = crop.size

    x1 = int(cw * epic_x_ratio)
    y1 = 10
    x2 = cw
    y2 = int(ch * epic_y_ratio)

    return crop.crop((x1, y1, x2, y2))


def relocate_epic_id_region(
    crop: Image.Image,
    epic_x_ratio: float = 0.60,
    epic_y_ratio: float = 0.25,
    bottom_empty_ratio: float = 0.30,
    padding: int = 6,
    bg_color: str = "white",
) -> Image.Image:
    """
    Extracts EPIC ID region from right side of voter crop,
    removes it, and pastes it into bottom empty area.
    """

    cw, ch = crop.size

    # -------------------------------
    # 1️⃣ Extract EPIC region
    # -------------------------------
    epic_region = extract_epic_region(crop, epic_x_ratio=epic_x_ratio, epic_y_ratio=epic_y_ratio)

    epic_w, epic_h = epic_region.size

    # 2️⃣ Remove EPIC from right side
    draw = ImageDraw.Draw(crop)
    draw.rectangle([int(cw * epic_x_ratio), 0, cw, int(ch * epic_y_ratio)], fill=bg_color)

    # 3️⃣ Compute bottom paste position
    bottom_start_y = int(ch * (1 - bottom_empty_ratio))

    paste_x = padding
    paste_y = bottom_start_y + padding

    # 4️⃣ Paste WITHOUT resizing (clip if overflow)
    if paste_y + epic_h <= ch:
        crop.paste(epic_region, (paste_x, paste_y))
    else:
        # Clip vertically if extreme edge case
        visible_h = ch - paste_y
        crop.paste(epic_region.crop((0, 0, epic_w, visible_h)), (paste_x, paste_y))

    return crop


def append_voter_end_marker(
    crop: Image.Image,
    marker_img: Image.Image,
    scale: float = 2.0,  # 🔥 OCR-critical tuning knob
    bottom_padding_px: int = 8,  # 🔥 distance from bottom edge
    left_padding_px: int = 8,  # safe left margin
) -> Image.Image:
    """
    Appends a scaled VOTER_END marker image at bottom-left of the crop.
    Marker is resized proportionally for OCR visibility.
    """

    cw, ch = crop.size
    mw, mh = marker_img.size

    # --- Resize marker proportionally ---
    new_mw = int(mw * scale)
    new_mh = int(mh * scale)

    marker_resized = marker_img.resize((new_mw, new_mh), Image.BICUBIC)

    # --- Safety check ---
    if new_mh + bottom_padding_px > ch:
        raise ValueError(
            f"Marker too tall ({new_mh}px) for crop height ({ch}px). " f"Reduce scale or padding."
        )

    # --- Paste position (bottom-left) ---
    paste_x = left_padding_px
    paste_y = ch - new_mh - bottom_padding_px

    # --- Work on a copy ---
    out = crop.copy()

    # --- Ensure white background for OCR contrast ---
    bg = Image.new("RGB", (new_mw, new_mh), "white")
    out.paste(bg, (paste_x, paste_y))
    out.paste(marker_resized, (paste_x, paste_y))

    return out


def crop_voter_boxes_dynamic(input_jpg):
    # os.makedirs(CROPS_DIR, exist_ok=True)
    lang = detect_ocr_language_from_filename(input_jpg)
    extract_street_info(input_jpg, lang=lang)

    img = Image.open(input_jpg)
    W, H = img.size

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
    photo_w_ratio = 380 / 1555
    photo_y_ratio = (620 - 480) / 620  # starting point

    count = 1
    crops = []

    for r in range(ROWS):
        for c in range(COLS):
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

            px_left = max(0, px_left - pad_x)
            px_top = max(0, px_top - pad_y)
            px_right = min(cw, px_right + pad_x)
            px_bottom = min(ch, px_bottom + pad_y)

            draw = ImageDraw.Draw(crop)
            draw.rectangle([px_left, px_top, px_right, px_bottom], fill="white")

            crop = relocate_epic_id_region(crop)
            crop = append_voter_end_marker(
                crop, marker_img=VOTER_END_MARKER, scale=2.0, left_padding_px=500
            )

            # append crop and associated file path to crops
            crops.append(
                {
                    "crop_name": f"{os.path.basename(input_jpg).replace('.jpg', '')}_voter_{count:02d}.jpg",
                    "crop": crop,
                    "lang": lang,
                }
            )

            # ocr_text = extract_text_from_image(f"{out_dir}/voter_{count:02d}.png")
            # epic_id = extract_epic_id(crop, count)
            # cleaned_data = clean_with_llm(ocr_text, epic_id)
            # append_to_csv(cleaned_data)
            # logger.info(f"Cleaned Data for voter_{count:02d}:\n{json.dumps(cleaned_data, indent=2)}")

            count += 1

    stacked_image = stack_voter_crops_vertically(crops)
    stacked_image.save(
        f"{CROPS_DIR}/{os.path.basename(input_jpg).replace('.jpg', '')}_stacked_crops.jpg"
    )

    stacked_ocr_text = extract_text_from_image(
        f"{CROPS_DIR}/{os.path.basename(input_jpg).replace('.jpg', '')}_stacked_crops.jpg",
        lang=lang,
    )

    # Save stacked OCR text to a file
    with open(
        f"{CROPS_DIR}/{os.path.basename(input_jpg).replace('.jpg', '')}_stacked_ocr.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(stacked_ocr_text)
    logger.info(
        f"Saved stacked OCR text: {os.path.basename(input_jpg).replace('.jpg', '')}_stacked_ocr.txt"
    )

    return crops


def crop_voter_boxes(png_dir: str, progress=None, limit=None):
    """
    Crops voter boxes from each page PNG and saves them to crops_dir
    """

    # Record the start time
    start_time = time.perf_counter()  # Or time.time() for less precision

    files = sorted(os.listdir(png_dir))
    if limit is not None:
        files = files[:limit]

    task = None
    if progress:
        task = progress.add_task("✂️ PNGs -> Crops", total=len(files) - 1)  # -1 to .DS_Store (MacOS)

    crops = []
    for file in files:
        if file.lower().endswith(".png"):
            if progress:
                progress.advance(task)

            input_png_path = os.path.join(png_dir, file)
            crops.extend(crop_voter_boxes_dynamic(input_png_path))

    # Record the end time
    end_time = time.perf_counter()  # Or time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    logger.info(f"Time taken by crop_voter_boxes: {elapsed_time:.3f} seconds.")
    return crops


def extract_street_info(input_jpg, lang):
    """
    Extracts street information from the top 5% area of the input page JPG using OCR.
    """

    img = Image.open(input_jpg)
    W, H = img.size
    top_area_height = int(H * 0.05)
    top_area = img.crop((0, 0, W, top_area_height))

    ocr_text = pytesseract.image_to_string(
        top_area, lang=lang, config="--psm 6"
    ).strip()  # Assume a single uniform block of text

    # Write the extracted text to a file
    street_info_file = input_jpg.replace(".jpg", "_street.txt")
    with open(street_info_file, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    return ocr_text


def crop_voter_boxes_parallel(jpg_dir: str, progress=None, max_workers=4, limit=None):
    """
    Crops voter boxes from each page JPG using multi-threading.
    """

    start_time = time.perf_counter()

    jpgs = sorted(f for f in os.listdir(jpg_dir) if f.lower().endswith(".jpg"))
    if limit is not None:
        jpgs = jpgs[:limit]

    task = None
    if progress:
        task = progress.add_task("✂️ JPGs → Crops", total=len(jpgs))

    crops = []

    def _crop_worker(jpg):
        input_jpg_path = os.path.join(jpg_dir, jpg)
        return crop_voter_boxes_dynamic(input_jpg_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(_crop_worker, jpg): jpg for jpg in jpgs}

        for future in as_completed(future_to_file):
            jpg = future_to_file[future]

            try:
                result = future.result()
                crops.extend(result)
                if progress and task:
                    progress.advance(task)
            except Exception as e:
                logger.error(f"❌ Failed cropping {jpg}: {e}")

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Time taken by crop_voter_boxes: {elapsed_time:.3f} seconds.")

    return crops


def stack_voter_crops_vertically(
    crops: list[dict], padding: int = 10, bg_color: str = "white"
) -> Image.Image:
    """
    Vertically stack voter crops into a single column image.

    Args:
        crops: list of dicts containing {"crop": PIL.Image}
        padding: vertical spacing between voters
        bg_color: background color

    Returns:
        PIL.Image of stacked voters
    """

    assert crops, "No crops provided"

    # Extract images
    images = [c["crop"] for c in crops]

    # Normalize widths (safety)
    max_width = max(img.width for img in images)

    normalized = []
    for img in images:
        if img.width != max_width:
            padded = Image.new("RGB", (max_width, img.height), bg_color)
            padded.paste(img, (0, 0))
            normalized.append(padded)
        else:
            normalized.append(img)

    total_height = sum(img.height for img in normalized) + padding * (len(normalized) - 1)

    stacked = Image.new("RGB", (max_width, total_height), bg_color)

    y_offset = 0
    for img in normalized:
        stacked.paste(img, (0, y_offset))
        y_offset += img.height + padding

    return stacked
