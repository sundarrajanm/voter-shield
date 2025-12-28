from utilities import normalize_tamil_text, parse_single_voter_ocr_tamil


def test_single_voter_tamil():
    text = """
    பெயர்‌ : ஆறுமுகம்‌ -
    தந்தையின்‌ பெயர்‌: நடேசநாயக்கர்‌ -
    வீட்டு எண்‌ :1
    வயது : 79 பாலினம்‌ : ஆண்‌
    4021150705 |
    """

    result = parse_single_voter_ocr_tamil(text)

    assert normalize_tamil_text(result["name"]) == normalize_tamil_text("ஆறுமுகம் -")
    assert normalize_tamil_text(result["father_name"]) == normalize_tamil_text("நடேசநாயக்கர் -")
    assert result["husband_name"] is None
    assert result["age"] == 79
    assert result["gender"] == "M"
