import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from csv_extract import parse_ocr_text

def test_basic_three_voters():
    text = """
Name : Rangaraj - Name : Lakshmi - Name : Kalaivani -
Father Name: Makali - Husband Name: Rangaraj - Father Name: Rangaraj -
House Number : 1 House Number : 1 House Number : 1
Age : 68 Gender : Male Age + 57 Gender : Female Age : 41 Gender : Female
"""
    voters = parse_ocr_text(text)

    assert len(voters) == 3
    assert voters[0]["father_name"] == "Makali -"
    assert voters[1]["husband_name"] == "Rangaraj -"
    assert voters[2]["father_name"] == "Rangaraj -"

def test_other_relationship():
    text = """
Name : Ramkumar -
Other: KARUPPAL
House Number : 9
Age : 52 Gender : Male
"""
    voters = parse_ocr_text(text)

    assert voters[0]["other_name"] == "KARUPPAL"
    assert voters[0]["father_name"] is None

def test_missing_lines_does_not_crash():
    text = "Name : Ram -"
    voters = parse_ocr_text(text)
    assert voters == []
