import pytesseract

file_name = ("../crops/2025-EROLLGEN-S22-116-FinalRoll-Revision1-ENG-244-WI_page_01_stacked_crops.jpg")
text = pytesseract.image_to_string(file_name, lang="eng", config="--psm 6 --oem 1")
lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

for line in lines:
    print(line)
