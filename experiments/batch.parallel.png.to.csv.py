import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pytesseract
import requests
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct" # "llama3.1:8b"   # change if needed
MAX_WORKERS = 1
BATCH_SIZE = 15
CROPS_DIR = "./crops"

def llm_batch_worker(args):
    """
    args = (batch_texts, batch_files)
    """
    batch_texts, batch_files = args

    results = clean_with_llm_batch(batch_texts)

    # Attach source_image here
    for r, fname in zip(results, batch_files, strict=False):
        r["source_image"] = fname

    return results

def clean_with_llm_batch(ocr_texts):
    """
    Cleans a batch of OCR voter texts using a local LLM.
    
    Args:
        ocr_texts (list[str]): Raw OCR text blocks
    
    Returns:
        list[dict]: Cleaned structured voter records
    """

    if not ocr_texts:
        return []

    system_prompt = (
        "You are a deterministic data-cleaning engine.\n\n"
        "Rules:\n"
        "- Do NOT guess missing values\n"
        "- Do NOT add new information\n"
        "- Use only what is explicitly present in the input text\n"
        "- If a field is missing, output null\n"
        "- Remove symbols like '-', '+', '=', '~'\n"
        "- Normalize names to Title Case\n"
        "- Normalize gender to Male/Female\n"
        "- Normalize house_no to alphanumeric and '/'\n"
        "- Age must be an integer or null\n\n"
        "Output rules:\n"
        "- Output STRICT JSON only\n"
        "- Output must be a JSON array\n"
        "- You MUST complete the JSON array and close all brackets\n"
        "- One output object per input item\n"
        "- Preserve input order exactly\n"
        "- No markdown, no explanations, no extra text\n"
    )

    user_prompt = (
        "Normalize the following OCR voter records.\n\n"
        "Each item is a raw OCR text block.\n\n"
        "Return a JSON array where each object has exactly these keys:\n"
        "{\n"
        '  "name": string | null,\n'
        '  "father_name": string | null,\n'
        '  "mother_name": string | null,\n'
        '  "husband_name": string | null,\n'
        '  "house_no": string | null,\n'
        '  "age": number | null,\n'
        '  "gender": "Male" | "Female" | null\n'
        "}\n\n"
        "INPUT:\n"
        + json.dumps(ocr_texts, ensure_ascii=False)
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "repeat_penalty": 1
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    raw_output = response.json()["response"].strip()

    try:
        cleaned = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM returned invalid JSON:\n{raw_output}"
        ) from e

    if not isinstance(cleaned, list):
        raise RuntimeError("Expected JSON array from LLM")

    if len(cleaned) != len(ocr_texts):
        raise RuntimeError("Output size mismatch with input batch")

    return cleaned


def extract_text_from_image(image_path: str) -> str:
    print(f"\n🔍 OCR processing: {image_path}")
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng", config="--psm 6 --oem 1")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    print(f"📄 OCR extracted {len(lines)} non-empty lines.")
    return "\n".join(lines)

def process_crops_folder_parallel(crops_dir=CROPS_DIR):
    crop_files = sorted(
        f for f in os.listdir(crops_dir)
        if f.lower().endswith(".png")
    )

    print(f"📁 Found {len(crop_files)} crop images")

    # 1️⃣ OCR phase (single process)
    ocr_texts = []
    for file in crop_files:
        path = os.path.join(crops_dir, file)
        ocr_texts.append(extract_text_from_image(path))

    # 2️⃣ Build batches
    batches = []
    for i in range(0, len(ocr_texts), BATCH_SIZE):
        batch_texts = ocr_texts[i:i + BATCH_SIZE]
        batch_files = crop_files[i:i + BATCH_SIZE]
        batches.append((batch_texts, batch_files))

    print(f"🚀 Total batches: {len(batches)}")

    # 3️⃣ Parallel LLM execution
    all_results = []
    total_start = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(llm_batch_worker, b) for b in batches]

        for future in as_completed(futures):
            result = future.result()
            all_results.extend(result)

    total_end = time.time()

    print("\n🎉 All processing completed")
    print(f"📊 Total records: {len(all_results)}")
    print(f"⏱️ Total time: {total_end - total_start:.2f} sec")

    # 4️⃣ Preserve original order
    all_results.sort(key=lambda x: x["source_image"])

    return all_results

if __name__ == "__main__":
    results = process_crops_folder_parallel()
    print(json.dumps(results[:5], indent=2))
