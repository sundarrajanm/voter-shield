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

    # # 3️⃣ OCR extraction
    # print("\n🔍 Step 3: OCR extraction")
    # ocr_results = extract_ocr_from_crops(CROPS_DIR)

    # # 3️⃣ OCR extraction
    print("\n🔍 Step 3: OCR extraction")
    ocr_results = extract_ocr_from_png(PNG_DIR, 50)

    # 4️⃣ CSV extraction
    print("\n🧠 Step 4: CSV extraction")
    cleaned_records = clean_and_extract_csv(ocr_results)
    
    # Write cleaned records to CSV
    write_final_csv(cleaned_records, CSV_DIR)

    # # Merge source_image back
    # for rec, meta in zip(cleaned_records, ocr_results):
    #     rec["source_image"] = meta["source_image"]

    # # 5️⃣ Serial number assignment
    # print("\n🔢 Step 5: Assigning serial numbers")
    # final_records = assign_serial_numbers(cleaned_records)

    # # 6️⃣ Write CSV
    # print("\n📊 Step 6: Writing final CSV")
    # write_final_csv(final_records, CSV_DIR)

    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()
