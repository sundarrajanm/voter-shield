import csv
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

INPUT_PDF = FIXTURES / "2025-EROLLGEN-S22-116-FinalRoll-Revision1-ENG-244-WI.pdf"
EXPECTED_CSV = FIXTURES / "expected_final_voter_data.csv"
GENERATED_CSV = ROOT / "csv" / "final_voter_data.csv"


def read_csv_as_list(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def normalize_rows(rows):
    """
    Normalize CSV rows to avoid false negatives:
    - Strip whitespace
    - Ensure stable key ordering
    """
    normalized = []
    for row in rows:
        normalized.append(
            {k: (v.strip() if isinstance(v, str) else v) for k, v in sorted(row.items())}
        )
    return normalized


def dict_field_diff(expected: dict, actual: dict):
    """
    Return a list of (field, expected_value, actual_value)
    only for fields that differ.
    """
    diffs = []

    all_keys = set(expected.keys()) | set(actual.keys())
    for key in sorted(all_keys):
        exp_val = expected.get(key)
        act_val = actual.get(key)

        if exp_val != act_val:
            diffs.append((key, exp_val, act_val))

    return diffs


@pytest.mark.regression
def test_pipeline_accuracy_regression(tmp_path):
    """
    Golden-file regression test:
    - Runs pipeline on known PDF
    - Compares generated CSV with reference CSV
    """

    # --- Step 1: Clean previous output ---
    if GENERATED_CSV.exists():
        GENERATED_CSV.unlink()

    # --- Step 2: Run pipeline ---
    result = subprocess.run(["python", "main.py", "--delete-old", "--regression"], check=True)

    assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"

    assert GENERATED_CSV.exists(), "Pipeline did not generate output CSV"

    # --- Step 3: Read CSVs ---
    expected_rows = read_csv_as_list(EXPECTED_CSV)
    actual_rows = read_csv_as_list(GENERATED_CSV)

    # --- Step 4: Compare row count ---
    assert len(actual_rows) == len(
        expected_rows
    ), f"Row count mismatch: expected {len(expected_rows)}, got {len(actual_rows)}"

    # --- Step 5: Compare content (field-level diff) ---
    failures = []

    field_counter = Counter()
    for i, (exp, act) in enumerate(zip(expected_rows, actual_rows, strict=False), start=1):
        if exp != act:
            field_diffs = dict_field_diff(exp, act)

            msg_lines = [f"Row {i} mismatch:"]
            for field, ev, av in field_diffs:
                field_counter[field] += 1
                msg_lines.append(f"  {field:<12} expected={repr(ev)}  actual={repr(av)}")

            failures.append("\n".join(msg_lines))

    if failures:
        print("\n\n".join(failures))
        print("\nField-level mismatch summary:")
        for field, count in field_counter.most_common():
            print(f"{field}: {count}")

        pytest.fail(f"{len(failures)} row(s) differ from baseline")
