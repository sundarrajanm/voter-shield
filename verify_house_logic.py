"""
Test script to verify the house number selection logic.
This demonstrates how the new logic works with various test cases.
"""

class MockOCRProcessor:
    """Mock OCR processor with just the helper methods for testing."""
    
    def _contains_numeric(self, text: str) -> bool:
        """Check if text contains any numeric characters."""
        if not text:
            return False
        return any(c.isdigit() for c in text)
    
    def _is_all_non_numeric(self, text: str) -> bool:
        """Check if text contains ONLY non-numeric characters (no digits at all)."""
        if not text or not text.strip():
            return False
        return not any(c.isdigit() for c in text)
    
    def apply_house_number_logic(self, ai_house: str, ocr_house: str) -> tuple[str, str]:
        """
        Apply the house number selection logic.
        
        Returns:
            (selected_value, reason)
        """
        # If AI value is empty, use OCR
        if not ai_house or not ai_house.strip():
            return (ocr_house, "AI empty, using OCR")
        
        # If OCR value is empty, use AI
        if not ocr_house or not ocr_house.strip():
            return (ai_house, "OCR empty, using AI")
        
        # NEW LOGIC: If AI has no numbers and OCR has numbers, prefer OCR
        should_use_ocr = (
            self._is_all_non_numeric(ai_house) and 
            self._contains_numeric(ocr_house)
        )
        
        if should_use_ocr:
            return (ocr_house, "AI has no numbers, OCR has numbers → prefer OCR")
        else:
            return (ai_house, "Default: prefer AI")


def test_house_number_logic():
    """Run test cases to verify the logic."""
    processor = MockOCRProcessor()
    
    test_cases = [
        # (ai_value, ocr_value, expected_result, description)
        ("2பீ", "2", "2பீ", "AI has number+Tamil, keep AI"),
        ("2C", "20", "2C", "AI has number+letter, keep AI"),
        ("ஏயூர்6", "5", "ஏயூர்6", "AI has Tamil+number, keep AI"),
        ("பஷ்ண", "13", "13", "AI has no numbers, OCR has numbers → prefer OCR"),
        ("ரஷ்ணம்", "14", "14", "AI has no numbers, OCR has numbers → prefer OCR"),
        ("பஷ்ண", "", "பஷ்ண", "OCR empty, keep AI even if no numbers"),
        ("", "15", "15", "AI empty, use OCR"),
        ("ABC", "16", "16", "AI has only letters, OCR has numbers → prefer OCR"),
        ("123", "456", "123", "Both have numbers, prefer AI (default)"),
        ("", "", "", "Both empty"),
    ]
    
    print("=" * 80)
    print("HOUSE NUMBER SELECTION LOGIC TEST")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for ai_val, ocr_val, expected, description in test_cases:
        result, reason = processor.apply_house_number_logic(ai_val, ocr_val)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status}")
        print(f"  Description: {description}")
        print(f"  AI:          '{ai_val}'")
        print(f"  OCR:         '{ocr_val}'")
        print(f"  Expected:    '{expected}'")
        print(f"  Got:         '{result}'")
        print(f"  Reason:      {reason}")
        print()
    
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = test_house_number_logic()
    exit(0 if success else 1)
