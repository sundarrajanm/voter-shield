import time
from PIL import Image, ImageDraw
import os
import pytesseract

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

def extract_text_from_image(image_path: str) -> str:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng", config="--psm 6 --oem 1")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

def extract_ocr_from_crops(crops_dir: str, limit=None):
    """
    Performs OCR on all cropped voter images.

    Returns:
        list[dict]: [
            {
                "source_image": "voter_01.png",
                "ocr_text": "Name : ...",
            },
            ...
        ]
    """
    results = []

    # Record the start time
    start_time = time.perf_counter() # Or time.time() for less precision

    serial_no = 1
    for file in sorted(os.listdir(crops_dir)):
        if file.lower().endswith(".png"):
            print(f"🔍 OCR processing: {file}")
            path = os.path.join(crops_dir, file)

            ocr_text = extract_text_from_image(path)
            epic_id = extract_epic_id(Image.open(path))
            print(f"📄 OCR extracted {len(ocr_text.splitlines())} non-empty lines.")

            if limit is not None and serial_no > limit:
                break

            if ocr_text.strip() != "":
                results.append({
                    "source_image": file,
                    "ocr_text": ocr_text,
                    "epic_id": epic_id,
                    "serial_no": serial_no,
                })
            serial_no += 1

    # Write results to ocr_results.json for inspection
    import json
    with open("ocr/ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Record the end time
    end_time = time.perf_counter() # Or time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Total number of OCR results:", len(results))
    
    print(f"Time taken by extract_ocr_from_crops: {elapsed_time:.3f} seconds.")
    return results

def extract_ocr_from_png(png_dir: str, batch_size):
    """
    Processes all voter png images in batches and returns cleaned results.
    """

    # 1️⃣ Collect & sort png files
    png_files = sorted(
        f for f in os.listdir(png_dir)
        if f.lower().endswith(".png")
    )

    print(f"📁 Found {len(png_files)} png images")

    all_results = []

    total_start = time.time()

    for idx, file in enumerate(png_files, start=1):
        if idx == batch_size + 1:
            print("\nReached batch size limit for testing. Stopping further processing.")
            break  # Limit to first `batch_size` files for testing

        path = os.path.join(png_dir, file)

        # 2️⃣ OCR extraction
        print(f"🔍 OCR processing: {file}")
        ocr_text = extract_text_from_image(path)
        all_results.append({
            "source_image": file,
            "ocr_text": ocr_text,
        })

    # Write results to ocr_results.json for inspection
    import json
    with open("ocr/ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    total_end = time.time()
    
    print(f"\nTotal time taken by extract_ocr_from_png: {total_end - total_start:.3f} seconds.\nReturning {len(all_results)} results.")

    print("\n🎉 All OCR processing completed")
    
    return all_results
