import time
import openai
import base64

from dotenv import load_dotenv
load_dotenv()

EXTRACTION_PROMPT = """
You are an expert at extracting structured data from Indian electoral roll pages.
Read the input image carefully and return ONLY a comma separated values (CSV) response where it 
should contain the following headers, with extracted data in each row:

serial, id, name, father_name, husband_name, mother_name, house_number, age, gender, assembly_number, assembly_name, section_no, section_name, part_no

Important:
- Return ONLY valid CSV output (no surrounding explanation).
- Always add the header row as the first line.
- Do not invent IDs or numbers.
- Use consistently M/F for gender.
- If any column is not present in the image, leave it empty.
- If a comma is part of any field, replace it with a space, so that the CSV format is not broken.
- And there should be no more than the above mentioned columns in the CSV.
- Do not expect - at the end of a field all the time. Sometimes fields may be separated by spaces or new lines.
- House number may be alphanumeric.
"""

def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def extract_to_csv_from_image(image_path):
    with open(image_path, "rb") as image_file:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"}},
                    ],
                }
            ]
        )
    extracted_text = response.choices[0].message.content
    return extracted_text

START_PAGE = 13
END_PAGE = 43
for i in range(START_PAGE, END_PAGE + 1):
    INPUT_IMAGE = f"./png/page_{i:02d}.png"
    OUTPUT_CSV = f"./csv/page_{i:02d}.csv"

    # Record the start time
    start_time = time.perf_counter() # Or time.time() for less precision

    print(f"🤖 Extracting CSV data from image: {INPUT_IMAGE}")
    csv_data = extract_to_csv_from_image(INPUT_IMAGE)
    save_text(OUTPUT_CSV, csv_data)
    print(f"📁 Saved extracted CSV to: {OUTPUT_CSV}")

    # Record the end time
    end_time = time.perf_counter() # Or time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Total time taken: {elapsed_time:.3f} seconds.")    

print("✅ CSV extraction completed!")
