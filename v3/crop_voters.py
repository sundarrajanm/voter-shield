import os
import time
from PIL import Image, ImageDraw
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed

import json

from logger import setup_logger
logger = setup_logger()

from PIL import Image, ImageDraw
import os

def crop_voter_columns_dynamic(png_dir, input_png_path):
    """
    Returns 3 column-wise crops per page PNG.
    Each crop contains up to 10 voters stacked vertically.
    """

    page_metadata = extract_header_footer_info(input_png_path)

    img = Image.open(input_png_path)
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
    col_w = content_w / COLS
    row_h = content_h / ROWS

    # Photo removal ratios (same as your logic)
    photo_w_ratio = 380 / 1555
    photo_y_ratio = (620 - 480) / 620

    crops = []

    for c in range(COLS):
        left = int(content_x + c * col_w)
        right = int(left + col_w)
        upper = int(content_y)
        lower = int(content_y + content_h)

        column_crop = img.crop((left, upper, right, lower))
        cw, ch = column_crop.size

        draw = ImageDraw.Draw(column_crop)

        # Remove photo region ROW BY ROW inside column
        for r in range(ROWS):
            row_top = int(r * row_h)
            row_bottom = int(row_top + row_h)

            px_left = int(cw * (1 - photo_w_ratio))
            px_top = int(row_top + row_h * photo_y_ratio)
            px_right = cw
            px_bottom = row_bottom

            # Padding
            pad_x = int(cw * 0.02)
            pad_y = int(row_h * 0.02)

            px_left = max(0, px_left - pad_x)
            px_top = max(0, px_top - pad_y)
            px_right = min(cw, px_right + pad_x)
            px_bottom = min(ch, px_bottom + pad_y)

            draw.rectangle(
                [px_left, px_top, px_right, px_bottom],
                fill="white"
            )
        
        # Save the crop and its metadata
        column_crop.save(f"{png_dir}/{os.path.basename(input_png_path).replace('.png', '')}_column_{c+1}.png")        

        crops.append({
            "crop_name": f"{os.path.basename(input_png_path).replace('.png', '')}_column_{c+1}.png",
            "crop": column_crop,
            "page_metadata": page_metadata
        })

    return crops

def crop_voter_boxes_dynamic(input_png):
    # os.makedirs(out_dir, exist_ok=True)
    extract_street_info(input_png)
    

    img = Image.open(input_png)

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

            px_left   = max(0, px_left - pad_x)
            px_top    = max(0, px_top - pad_y)
            px_right  = min(cw, px_right + pad_x)
            px_bottom = min(ch, px_bottom + pad_y)

            draw = ImageDraw.Draw(crop)
            draw.rectangle([px_left, px_top, px_right, px_bottom], fill="white")

            # crop.save(f"{out_dir}/{os.path.basename(input_png).replace('.png', '')}_voter_{count:02d}.png")

            # append crop and associated file path to crops
            crops.append({
                "crop_name": f"{os.path.basename(input_png).replace('.png', '')}_voter_{count:02d}.png",
                "crop": crop
            })

            # ocr_text = extract_text_from_image(f"{out_dir}/voter_{count:02d}.png")
            # epic_id = extract_epic_id(crop, count)
            # cleaned_data = clean_with_llm(ocr_text, epic_id)
            # append_to_csv(cleaned_data)
            # logger.info(f"Cleaned Data for voter_{count:02d}:\n{json.dumps(cleaned_data, indent=2)}")

            count += 1
    return crops

def crop_voter_boxes(png_dir: str, progress=None, limit=None):
    """
    Crops voter boxes from each page PNG and saves them to crops_dir
    """

    # Record the start time
    start_time = time.perf_counter() # Or time.time() for less precision

    files = sorted(os.listdir(png_dir))
    if limit is not None:
        files = files[:limit]

    task = None
    if progress:
        task = progress.add_task(f"✂️ PNGs -> Crops", total=len(files) - 1) # -1 to .DS_Store (MacOS)

    crops = []
    for file in files:
        if file.lower().endswith(".png"):
            if progress:
                progress.advance(task)

            input_png_path = os.path.join(png_dir, file)
            crops.extend(crop_voter_boxes_dynamic(input_png_path))

    # Record the end time
    end_time = time.perf_counter() # Or time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    logger.info(f"Time taken by crop_voter_boxes: {elapsed_time:.3f} seconds.")
    return crops

def extract_street_info(input_png):
    """
    Extracts street information from the top 5% area of the input page PNG using OCR.
    """

    img = Image.open(input_png)
    W, H = img.size
    top_area_height = int(H * 0.05)
    top_area = img.crop((0, 0, W, top_area_height))

    ocr_text = pytesseract.image_to_string(
        top_area,
        lang='eng',
        config='--psm 6').strip()  # Assume a single uniform block of text
    
    # Write the extracted text to a file
    street_info_file = input_png.replace('.png', '_street.txt')
    with open(street_info_file, 'w', encoding='utf-8') as f:
        f.write(ocr_text)

    return ocr_text

import re
from PIL import Image
import pytesseract

def extract_street(header_text: str) -> str | None:
    """
    Extract street name robustly:
    - Starts after 'Section No and Name'
    - Continues even if 'Ward No <n>' appears in the middle
    - Stops only when OCR noise starts
    """

    # Noise / termination markers
    noise_markers = r"(?:\s+—|\s+\||\s+\[|\s+Name\s*:|$)"

    m = re.search(
        rf"Section\s+No\s+and\s+Name\s+(.+?){noise_markers}",
        header_text,
        flags=re.IGNORECASE
    )

    return m.group(1).strip() if m else None

def extract_header_footer_info(input_png: str) -> dict:
    """
    Extracts street, age_as_on, and published_on from page PNG.
    """

    img = Image.open(input_png)
    W, H = img.size

    # --- Header & Footer crops ---
    header_h = int(H * 0.06)      # top 6%
    footer_h = int(H * 0.06)      # bottom 6%

    header = img.crop((0, 0, W, header_h))
    footer = img.crop((0, H - footer_h, W, H))

    # --- OCR ---
    header_text = pytesseract.image_to_string(
        header, lang="eng", config="--psm 6"
    )

    footer_text = pytesseract.image_to_string(
        footer, lang="eng", config="--psm 6"
    )

    # Normalize whitespace for regex safety
    header_text = " ".join(header_text.split())
    footer_text = " ".join(footer_text.split())

    result = {
        "header_text": header_text,
        "footer_text": footer_text,
        "street": None,
        "age_as_on": None,
        "published_on": None
    }

    # # --- STREET ---
    # # Anchor street ending strictly at W.No.<number>
    # m = re.search(
    #     r"Section\s+No\s+and\s+Name\s+(.+?\bW\.No\.?\s*\d+)",
    #     header_text,
    #     re.IGNORECASE
    # )

    # if m:
    #     result["street"] = m.group(1).strip()

    result["street"] = extract_street(header_text)

    # --- AGE AS ON ---
    m = re.search(
        r"Age\s+as\s+on\s+(\d{2}-\d{2}-\d{4})",
        footer_text,
        re.IGNORECASE
    )
    if m:
        result["age_as_on"] = m.group(1)


    # --- DATE OF PUBLICATION ---
    m = re.search(
        r"Publication\s*[:\-–]*\s*(\d{2}-\d{2}-\d{4})",
        footer_text,
        re.IGNORECASE
    )

    if m:
        result["published_on"] = m.group(1)

    # Write the extracted result into json file
    street_info_file = input_png.replace('.png', '_header_footer.json')
    with open(street_info_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return result

def crop_voter_boxes_parallel(
    png_dir: str,
    progress=None,
    max_workers=4,
    limit=None
):
    """
    Crops voter boxes from each page PNG using multi-threading.
    """

    start_time = time.perf_counter()

    files = sorted(f for f in os.listdir(png_dir) if f.lower().endswith(".png"))
    if limit is not None:
        files = files[:limit]

    task = None
    if progress:
        task = progress.add_task(
            "✂️ PNGs → Crops",
            total=len(files)
        )

    crops = []

    def _crop_worker(file):
        input_png_path = os.path.join(png_dir, file)
        return crop_voter_columns_dynamic(png_dir, input_png_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_crop_worker, file): file
            for file in files
        }

        for future in as_completed(future_to_file):
            file = future_to_file[future]

            try:
                result = future.result()
                crops.extend(result)
            except Exception as e:
                logger.error(f"❌ Failed cropping {file}: {e}")
            finally:
                if progress and task:
                    progress.advance(task)

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Time taken by crop_voter_boxes: {elapsed_time:.3f} seconds.")

    return crops