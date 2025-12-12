import time
import requests
import json
from PIL import Image
import pytesseract
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct" # "llama3.1:8b"   # change if needed

def extract_text_from_image(image_path: str) -> str:
    print(f"\n🔍 OCR processing: {image_path}")
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng", config="--psm 6 --oem 1")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    print(f"📄 OCR extracted {len(lines)} non-empty lines.")
    return "\n".join(lines)

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
        "prompt": system_prompt + "\n\n" + user_prompt,
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

CROPS_DIR = "./crops"
BATCH_SIZE = 20   # start with 20, you can tune later


def process_crops_folder(crops_dir=CROPS_DIR, batch_size=BATCH_SIZE):
    """
    Processes all voter crop images in batches and returns cleaned results.
    """

    # 1️⃣ Collect & sort crop files
    crop_files = sorted(
        f for f in os.listdir(crops_dir)
        if f.lower().endswith(".png")
    )

    print(f"📁 Found {len(crop_files)} crop images")

    all_results = []
    batch = []
    batch_files = []

    total_start = time.time()

    for idx, file in enumerate(crop_files, start=1):
        path = os.path.join(crops_dir, file)

        # 2️⃣ OCR extraction
        ocr_text = extract_text_from_image(path)
        batch.append(ocr_text)
        batch_files.append(file)

        # 3️⃣ When batch is full → call LLM
        if len(batch) == batch_size:
            print(f"\n🚀 Processing batch {len(all_results) + 1}–{len(all_results) + len(batch)}")
            start = time.time()

            results = clean_with_llm_batch(batch)

            end = time.time()
            print(f"✅ Batch completed in {end - start:.2f} sec")

            # 4️⃣ Attach source_file info (important later)
            for r, fname in zip(results, batch_files):
                r["source_image"] = fname
                all_results.append(r)

            batch.clear()
            batch_files.clear()

    # 5️⃣ Process remaining items
    if batch:
        print(f"\n🚀 Processing final batch ({len(batch)} items)")
        start = time.time()

        results = clean_with_llm_batch(batch)

        end = time.time()
        print(f"✅ Final batch completed in {end - start:.2f} sec")

        for r, fname in zip(results, batch_files):
            r["source_image"] = fname
            all_results.append(r)

    total_end = time.time()

    print("\n🎉 All processing completed")
    print(f"📊 Total records: {len(all_results)}")
    print(f"⏱️ Total time: {total_end - total_start:.2f} sec")

    return all_results

if __name__ == "__main__":
    results = process_crops_folder()

    # sanity check
    print(json.dumps(results, indent=2))
