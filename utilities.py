import re

ZERO_WIDTH_RE = re.compile(r"[\u200c\u200d\ufeff]")  # ZWNJ, ZWJ, BOM
NBSP_RE = re.compile(r"[\u00a0]")  # non-breaking space


def normalize_tamil_text(text: str) -> str:
    """
    Normalize Tamil OCR text WITHOUT collapsing newlines.
    - Removes zero-width unicode chars that break regex
    - Normalizes weird spaces, but preserves line boundaries
    """
    if not text:
        return text

    # Normalize newlines first
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove zero-width chars
    text = ZERO_WIDTH_RE.sub("", text)

    # Normalize NBSP to normal space
    text = NBSP_RE.sub(" ", text)

    # Clean up whitespace per line (preserve '\n')
    lines = []
    for ln in text.split("\n"):
        # collapse runs of spaces/tabs within the line only
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        lines.append(ln)

    # Keep non-empty lines (optional: if you want to preserve blank lines, remove the if)
    return "\n".join([ln for ln in lines if ln])


def parse_page_metadata_tamil(ocr_text: str) -> dict[str, str | None]:
    ocr_text = normalize_tamil_text(ocr_text)

    result = {"assembly": None, "part_no": None, "street": None}
    if not ocr_text:
        return result

    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return result

    line1, line2 = lines[0], lines[1]

    # Assembly: between ':' and 'Part'
    m_assembly = re.search(r"பெயர்\s*:\s*(.+?)\s+பாகம்", line1, re.UNICODE)
    if m_assembly:
        result["assembly"] = m_assembly.group(1).strip()

    # Part No
    m_part = re.search(r"பாகம்\s*எண்\.?\s*[:\-]?\s*(\d+)", line1, re.I)
    if m_part:
        result["part_no"] = int(m_part.group(1))

    # Street: keep the "1-" prefix (or any section prefix)
    # Example: "Section No and Name 1-Karupparayan Kovil Street Ward No-9"
    m_street = re.search(r"பிரிவு\s*எண்.*?பெயர்\s*[:\-]?\s*(.+)$", line2, re.I)
    if m_street:
        result["street"] = m_street.group(1).strip()

    return result


def replace_noise_words_with_corrections(ocr_text, corrections):
    cleaned = ocr_text
    for wrong, right in corrections.items():
        cleaned = cleaned.replace(wrong, right)
    return cleaned


def normalize_epic_candidate(token: str) -> str:
    """
    Normalize common OCR confusions for EPIC IDs.
    Applied only to EPIC candidates.
    """
    token = token.upper()

    # Common OCR confusions
    if token.startswith("V"):  # Y often read as V/v
        token = "Y" + token[1:]

    token = token.replace("O", "0")
    token = token.replace("I", "1")

    return token


def parse_single_voter_ocr_tamil(ocr_text: str) -> dict[str, str | None]:
    """
    Parse OCR text of a SINGLE Tamil voter box into structured fields.
    """

    result = {
        "epic_id": None,
        "name": None,
        "father_name": None,
        "husband_name": None,
        "mother_name": None,
        "other_name": None,
        "house_no": None,
        "age": None,
        "gender": None,
    }

    # Placeholder corrections (will be refined later)
    corrections = {
        "Narne": "Name",
        "Narme": "Name",
        "Narnme": "Name",
        "Famale": "Female",
        "Gander": "Gender",
    }

    # --- Normalize Tamil OCR ---
    ocr_text = normalize_tamil_text(ocr_text)
    ocr_text = replace_noise_words_with_corrections(ocr_text, corrections)

    if not ocr_text or "பெயர்" not in ocr_text:
        return result

    # Drop noise before first Name anchor
    ocr_text = ocr_text[ocr_text.find("பெயர்") :]

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    epic_candidates: list[str] = []

    for line in lines:
        # --- NAME (Tamil OCR tolerant; only once) ---
        if result["name"] is None:
            m = re.search(r"பெயர்\s*(?:[:=])?\s*(.+)$", line)
            if m:
                # Exclude relation lines that also contain 'பெயர்'
                if not any(rel in line for rel in ("தந்தை", "தாய்", "கணவர்", "மற்றவர்")):
                    result["name"] = m.group(1).strip()
                    continue

        # --- FATHER ---
        m = re.search(r"தந்தை.*?பெயர்\s*[:=]\s*(.+)$", line)
        if m:
            result["father_name"] = m.group(1).strip()
            continue

        # --- MOTHER ---
        m = re.search(r"தாய்.*?பெயர்\s*[:=]\s*(.+)$", line)
        if m:
            result["mother_name"] = m.group(1).strip()
            continue

        # --- HUSBAND ---
        m = re.search(r"கணவர்.*?பெயர்\s*[:=]\s*(.+)$", line)
        if m:
            result["husband_name"] = m.group(1).strip()
            continue

        # --- OTHER RELATION ---
        m = re.search(r"மற்றவர்\s*[:=]\s*(.+)$", line)
        if m:
            result["other_name"] = m.group(1).strip()
            continue

        # --- HOUSE NUMBER ---
        m = re.search(r"வீட்டு\s*எண்\s*[:=]\s*(.+)$", line)
        if m:
            result["house_no"] = m.group(1).strip()
            continue

        # --- AGE + GENDER ---
        if "வயது" in line and "பாலினம்" in line:
            age_m = re.search(r"வயது\s*[:+]\s*(\d+)", line)
            gender_m = re.search(r"பாலினம்\s*[:+]\s*(ஆண்|பெண்)", line)

            if age_m:
                result["age"] = int(age_m.group(1))
            if gender_m:
                result["gender"] = "M" if gender_m.group(1) == "ஆண்" else "F"
            continue

        # --- EPIC ID CANDIDATE ---
        raw = re.sub(r"[^A-Za-z0-9]", "", line)
        if not raw or len(raw) < 8:
            continue

        token = normalize_epic_candidate(raw)

        # Strict EPIC shape: letters + digits
        # Typical Indian EPICs: 2–4 letters + 6–8 digits
        if re.fullmatch(r"[A-Z]{2,4}[0-9]{6,8}", token):
            epic_candidates.append(token)

    # Pick the most reliable EPIC (more digits preferred)
    if epic_candidates:

        def score(epic: str) -> tuple[int, int]:
            return (sum(c.isdigit() for c in epic), len(epic))

        result["epic_id"] = max(epic_candidates, key=score)

    return result


VOTER_END_TOKEN = "VOTER_END"


def split_voters_from_page_ocr(page_ocr_text: str) -> list[str]:
    """
    Split page-level OCR text into individual voter OCR blocks
    using ONLY the VOTER_END token as the delimiter.
    """

    text = page_ocr_text.replace("\r", "").strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    voters: list[str] = []
    buf: list[str] = []

    for line in lines:
        # Collect everything
        if VOTER_END_TOKEN in line:
            # remove the marker before appending
            cleaned_line = line.replace(VOTER_END_TOKEN, "").strip()
            if cleaned_line:
                buf.append(cleaned_line)

            voter_text = "\n".join(buf).strip()
            if voter_text:
                voters.append(voter_text)

            buf = []  # reset buffer
        else:
            buf.append(line)

    # ⚠️ Safety: drop trailing incomplete voter (no VOTER_END)
    return voters
