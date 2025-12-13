from pdf2image import convert_from_path
from PIL import Image
import time
import pytesseract

INPUT_PDF = "./pdf/2025-EROLLGEN-S22-116-FinalRoll-Revision1-ENG-244-WI.pdf"
OUTPUT_PNG_DIR = "png"
OUTPUT_OCR_DIR = "ocr"

def extract_text_from_image(image_path: str) -> str:
    print(f"\n🔍 OCR processing: {image_path}")
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng", config="--psm 6 --oem 1")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    print(f"📄 OCR extracted {len(lines)} non-empty lines.")
    return "\n".join(lines)

# Record the start time
start_time = time.perf_counter() # Or time.time() for less precision

print("🚀 Converting PDF to PNG images...  ")
pages = convert_from_path(INPUT_PDF, dpi=450)

START_PAGE = 3
END_PAGE = 43

for i, page in enumerate(pages):
    if i >= START_PAGE - 1 and i < END_PAGE:
        print(f"🤖 Writing page {i+1:02d} as image.")
        page.save(f"./{OUTPUT_PNG_DIR}/page_{i+1:02d}.png", "PNG")
        # img = Image.open(f"./{OUTPUT_PNG_DIR}/page_{i+1:02d}.png")
        ocr_text = extract_text_from_image(f"./{OUTPUT_PNG_DIR}/page_{i+1:02d}.png")
        with open(f"./{OUTPUT_OCR_DIR}/page_{i+1:02d}.txt", "w") as f:
            f.write(ocr_text)

# Record the end time
end_time = time.perf_counter() # Or time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Total time taken: {elapsed_time:.3f} seconds.")
print("✅ PDF to PNG to OCR conversion completed.")
