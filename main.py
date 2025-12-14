# main.py

import os
from config import (
    PDF_DIR, PNG_DIR, CROPS_DIR, CSV_DIR, DPI
)

from pdf_to_png import convert_pdfs_to_png
from crop_voters import crop_voter_boxes
from ocr_extract import extract_ocr_from_crops, extract_ocr_from_png
from csv_extract import clean_and_extract_csv
from write_csv import write_final_csv
# from llm_cleaner import clean_with_llm_batch
# from serial_number import assign_serial_numbers
# from csv_writer import write_final_csv

def main():
    print("🛡️ VoterShield Pipeline Started")

    # # 1️⃣ PDF → PNG
    # print("\n📄 Step 1: Converting PDFs to PNGs")
    # convert_pdfs_to_png(PDF_DIR, PNG_DIR, DPI)

    # # 2️⃣ Crop voter boxes
    # print("\n✂️ Step 2: Cropping voter boxes")
    # crop_voter_boxes(PNG_DIR, CROPS_DIR)

    # 3️⃣ OCR extraction
    print("\n🔍 Step 3: OCR extraction")
    ocr_results = extract_ocr_from_crops(CROPS_DIR)

    # 4️⃣ CSV extraction
    print("\n🧠 Step 4: CSV extraction")
    import json
    with open("ocr/ocr_results.json", "r", encoding="utf-8") as f:
        ocr_results = json.load(f)
    cleaned_records = clean_and_extract_csv(ocr_results)
    
    # # 6️⃣ Write CSV
    write_final_csv(cleaned_records, CSV_DIR)
    # print("\n📊 Step 6: Writing final CSV")

    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
