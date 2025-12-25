import csv
import os


def write_final_csv(cleaned_records, CSV_DIR):
    """
    Writes the cleaned voter records to a final CSV file.
    """

    csv_path = f"{CSV_DIR}/final_voter_data.csv"
    os.makedirs(CSV_DIR, exist_ok=True)

    all_fieldnames = set()
    for record in cleaned_records:
        all_fieldnames.update(record.keys())

    preferred_order = [
        "assembly",
        "part_no",
        "street",
        "serial_no",
        "epic_id",
        "name",
        "father_name",
        "mother_name",
        "husband_name",
        "other_name",
        "house_no",
        "age",
        "gender",
    ]

    filtered_columns = ["source_image", "ocr_text", "doc_id", "page_no", "voter_no"]

    fieldnames = [f for f in preferred_order if f in all_fieldnames] + [
        f for f in all_fieldnames if f not in preferred_order and f not in filtered_columns
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, extrasaction="ignore"  # optional safety
        )
        writer.writeheader()
        for record in cleaned_records:
            writer.writerow(record)

    print(f"✅ Final CSV (voters: {len(cleaned_records)}) written to {csv_path}")
