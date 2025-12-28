from utilities import normalize_tamil_text, parse_page_metadata_tamil


def test_parse_page_metadata_tamil():
    # Read sample OCR text from sample_tamil_ocr.txt
    with open(
        "tests/fixtures/2025-EROLLGEN-S22-114-FinalRoll-Revision1-TAM-1-WI_page_01_street.txt",
        encoding="utf-8",
    ) as f:
        text = f.read()

    metadata = parse_page_metadata_tamil(text)

    assert metadata["assembly"] == normalize_tamil_text("114-திருப்பூர்‌ (தெற்கு)")
    assert metadata["part_no"] == 1
    assert metadata["street"] == normalize_tamil_text(
        "1-திருப்பூர்‌ (மா), முருங்கபாளையம்‌ 1து தெரு வார்டு எண்‌.27"
    )
