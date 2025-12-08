import argparse
import os
import time
import pandas as pd
import math

# -------------- Configurable bits --------------

COMMON_GENERIC_NAMES = {
    "kumar", "raja", "raju", "selvi", "lakshmi", "laxmi", "devi",
    "babu", "mani", "manikandan", "suresh", "ramesh"
}

YOUTH_MIN_AGE = 18
YOUTH_MAX_AGE = 22
YOUTH_PER_HOUSE_THRESHOLD = 5  # "More than 4" from rule

AGE_CLUSTER_THRESHOLD = 15  # Rule 8: same age appears > 15 times in a part

ADULT_MIN_AGE_MISSING_REL = 25  # Rule 5: adult with no relations


# -------------- Helpers --------------

def norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def to_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return math.nan

def add_flag(df, idx, reason):
    """Mark a row as suspicious and append reason text."""
    df.at[idx, "suspicious"] = True
    reasons = df.at[idx, "reasons"]
    reasons.append(reason)
    # Print a concise line for this flagged voter
    print(f"  ⚠️ Suspicious [serial={df.at[idx, 'serial']}, "
          f"name={df.at[idx, 'name']}, house={df.at[idx, 'house_number']}, id={df.at[idx, 'id']}] → {reason}")


# -------------- Rules Implementation --------------

def run_rule_1_same_name_house_age_spread(df):
    """
    Rule 1:
    Duplicate Names + Same House Number + Big Age Gap.
    Flag when same name & house_number but age spread > 20.
    """
    print("\n▶ Running Rule 1: same name + same house_number with large age difference...")
    grouped = df.groupby(["name_norm", "house_norm"])
    for (name_norm, house_norm), idxs in grouped.groups.items():
        if name_norm == "" or house_norm == "":
            continue
        ages = df.loc[idxs, "age_num"].dropna()
        if len(ages) > 1:
            spread = ages.max() - ages.min()
            if spread > 20:
                for idx in idxs:
                    add_flag(df, idx, f"Rule1: same name & house with age spread {int(ages.min())}-{int(ages.max())}")


def run_rule_2_duplicate_or_missing_id(df):
    """
    Rule 2:
    Duplicate EPIC ID or missing ID.
    """
    print("\n▶ Running Rule 2: duplicate or missing voter ID...")
    id_clean = df["id_clean"]

    # Missing IDs
    missing_mask = (id_clean == "")
    for idx in df[missing_mask].index:
        add_flag(df, idx, "Rule2: missing voter ID")

    # Duplicate IDs
    dup_mask = id_clean.duplicated(keep=False) & (id_clean != "")
    for idx in df[dup_mask].index:
        add_flag(df, idx, "Rule2: duplicate voter ID")


def run_rule_3_duplicate_full_profile(df):
    """
    Rule 3:
    Two people with identical demographic profile.
    (name, father_name, husband_name, age, gender, house_number)
    """
    print("\n▶ Running Rule 3: exact duplicate demographic profile...")
    grouped = df.groupby(
        ["name_norm", "father_norm", "husband_norm", "house_norm", "gender_norm", "age_num"]
    )
    for key, idxs in grouped.groups.items():
        if len(idxs) > 1:
            for idx in idxs:
                add_flag(df, idx, "Rule3: duplicate full demographic profile")


def run_rule_4_sequential_ids_in_house(df):
    """
    Rule 4:
    Multiple voters with suspiciously sequential EPIC IDs inside same house.
    We look for runs of >=3 IDs with same prefix and consecutive numeric suffix.
    """
    print("\n▶ Running Rule 4: sequential EPIC IDs within same house...")

    def split_id(epic):
        # Example: ZNC2208767 -> ("ZNC", 2208767)
        s = epic
        if not s:
            return None, None
        prefix = ""
        num_part = ""
        for ch in s:
            if ch.isdigit():
                num_part += ch
            else:
                prefix += ch
        if num_part == "":
            return None, None
        try:
            return prefix, int(num_part)
        except ValueError:
            return None, None

    for house_val, idxs in df.groupby("house_norm").groups.items():
        if not house_val:
            continue
        # Build list of (idx, prefix, num)
        rows = []
        for idx in idxs:
            epic = df.at[idx, "id_clean"]
            prefix, num = split_id(epic)
            if prefix is None or num is None:
                continue
            rows.append((idx, prefix, num))
        if not rows:
            continue

        # Group by prefix
        from collections import defaultdict
        by_prefix = defaultdict(list)
        for idx, prefix, num in rows:
            by_prefix[prefix].append((idx, num))

        for prefix, items in by_prefix.items():
            # sort by numeric suffix
            items.sort(key=lambda x: x[1])
            # find runs of length >=3 with step=1
            run = [items[0]]
            for current in items[1:]:
                if current[1] == run[-1][1] + 1:
                    run.append(current)
                else:
                    if len(run) >= 3:
                        for idx_run, num_run in run:
                            add_flag(df, idx_run,
                                     f"Rule4: sequential EPIC IDs in house ({prefix}{run[0][1]}..{run[-1][1]})")
                    run = [current]
            if len(run) >= 3:
                for idx_run, num_run in run:
                    add_flag(df, idx_run,
                             f"Rule4: sequential EPIC IDs in house ({prefix}{run[0][1]}..{run[-1][1]})")


def run_rule_5_adults_missing_relations(df):
    """
    Rule 5:
    Adults (age >= ADULT_MIN_AGE_MISSING_REL) with no father/husband/mother name.
    """
    print("\n▶ Running Rule 5: adults with all relation names missing...")
    mask = (
        (df["age_num"] >= ADULT_MIN_AGE_MISSING_REL) &
        (df["father_norm"] == "") &
        (df["husband_norm"] == "") &
        (df["mother_norm"] == "")
    )
    for idx in df[mask].index:
        add_flag(df, idx, "Rule5: adult with no father/husband/mother name")


def run_rule_6_many_youth_in_house(df):
    """
    Rule 6:
    Too many youth (18–22) in same house.
    """
    print(f"\n▶ Running Rule 6: houses with >= {YOUTH_PER_HOUSE_THRESHOLD} voters aged {YOUTH_MIN_AGE}-{YOUTH_MAX_AGE}...")
    grouped = df.groupby("house_norm")
    for house_val, idxs in grouped.groups.items():
        ages = df.loc[idxs, "age_num"]
        youth_mask = (ages >= YOUTH_MIN_AGE) & (ages <= YOUTH_MAX_AGE)
        if youth_mask.sum() >= YOUTH_PER_HOUSE_THRESHOLD:
            for idx in ages[youth_mask].index:
                add_flag(df, idx,
                         f"Rule6: house has {youth_mask.sum()} voters aged {YOUTH_MIN_AGE}-{YOUTH_MAX_AGE}")


def run_rule_7_generic_name_no_relations(df):
    """
    Rule 7:
    Very generic name, with missing relations.
    """
    print("\n▶ Running Rule 7: generic names with no relation names...")
    for idx, row in df.iterrows():
        if row["name_norm"] in COMMON_GENERIC_NAMES:
            if (row["father_norm"] == "" and
                row["husband_norm"] == "" and
                row["mother_norm"] == ""):
                add_flag(df, idx, "Rule7: generic name with no relations given")


# def run_rule_8_age_repeated_in_part(df):
#     """
#     Rule 8:
#     Age repeated too often in same part (assembly + part_no).
#     Flag if count(age) > AGE_CLUSTER_THRESHOLD.
#     """
#     print(f"\n▶ Running Rule 8: ages overly frequent in same part (>{AGE_CLUSTER_THRESHOLD})...")
#     grouped = df.groupby(["assembly_number", "part_no", "age_num"])
#     for (assembly, part, age), idxs in grouped.groups.items():
#         if pd.isna(age):
#             continue
#         if len(idxs) > AGE_CLUSTER_THRESHOLD:
#             for idx in idxs:
#                 add_flag(df, idx,
#                          f"Rule8: age {int(age)} appears {len(idxs)} times in same part {assembly}-{part}")

def run_rule_8_age_repeated_in_part(df):
    """
    Rule 8:
    Age repeated too often in same part + street.
    Group by (assembly_number, part_no, section_no, section_name, age_num)
    Flag if count(age) > AGE_CLUSTER_THRESHOLD.
    """
    print(f"\n▶ Running Rule 8: ages overly frequent in same street part (>{AGE_CLUSTER_THRESHOLD})...")

    grouped = df.groupby([
        "assembly_number",
        "part_no",
        "section_no",
        "section_name",
        "age_num"
    ])

    for (assembly, part, sec_no, sec_name, age), idxs in grouped.groups.items():
        if pd.isna(age):
            continue
        if len(idxs) > AGE_CLUSTER_THRESHOLD:
            for idx in idxs:
                add_flag(df, idx,
                         f"Rule8: age {int(age)} appears {len(idxs)} times in {sec_name} (section {sec_no}), part {part}")



def run_rule_9_serial_anomalies(df):
    """
    Rule 9:
    Serial number anomalies: duplicates or not strictly increasing.
    """
    print("\n▶ Running Rule 9: serial number anomalies...")

    # Duplicates
    serials = df["serial_num"]
    dup_mask = serials.duplicated(keep=False) & serials.notna()
    for idx in df[dup_mask].index:
        add_flag(df, idx, "Rule9: duplicate serial number")

    # Non-increasing sequence (based on row order)
    prev_serial = None
    prev_idx = None
    for idx, s in serials.items():
        if pd.isna(s):
            continue
        if prev_serial is not None and s <= prev_serial:
            add_flag(df, idx,
                     f"Rule9: serial {int(s)} not greater than previous serial {int(prev_serial)} at row {prev_idx}")
        prev_serial = s
        prev_idx = idx


def run_rule_10_orphan_voter(df):
    """
    Rule 10:
    Detect 'orphan voters' who do not connect to local family relationships.
    Based on Tamil Nadu naming patterns.
    """

    print("\n▶ Running Rule 10: orphan voters in street (no family linkage)...")

    # Group by street (section_no + section_name)
    grouped = df.groupby(["section_no", "section_name"])

    for (sec_no, sec_name), idxs in grouped.groups.items():
        sub = df.loc[idxs]

        # Build quick lookup sets
        father_names = set(sub["father_norm"].dropna())
        husband_names = set(sub["husband_norm"].dropna())

        for idx, row in sub.iterrows():
            age = row["age_num"]
            gender = row["gender_norm"]
            fname = row["father_norm"]
            hname = row["husband_norm"]

            # FEMALE: Expected husband for 30+
            if gender == "female" and age >= 30:
                if hname == "" or hname not in father_names and hname not in sub["name_norm"].values:
                    add_flag(df, idx,
                             f"Rule10: female {age}+ with no matching husband in same street {sec_name}")
                    continue

            # FEMALE <30: Expected father linkage
            if gender == "female" and age < 30:
                if fname == "" or fname not in sub["name_norm"].values:
                    add_flag(df, idx,
                             f"Rule10: female <30 with no matching father in same street {sec_name}")
                    continue

            # MALE: father linkage expected
            if gender == "male":
                if fname == "" or fname not in father_names and fname not in sub["name_norm"].values:
                    add_flag(df, idx,
                             f"Rule10: male voter with no matching father linkage in same street {sec_name}")

def run_rule_11_large_households(df, threshold=10):
    """
    Rule 11:
    More than `threshold` people in the same house_number + street (section_no + section_name).
    """
    print(f"\n▶ Running Rule 11: extremely large households (>{threshold})...")

    grouped = df.groupby(["section_no", "section_name", "house_norm"])

    for (sec_no, sec_name, house), idxs in grouped.groups.items():
        if house == "":
            continue

        count = len(idxs)
        if count > threshold:
            for idx in idxs:
                add_flag(df, idx,
                         f"Rule11: {count} residents in house {house}, street {sec_name} (section {sec_no})")

# -------------- Main pipeline --------------

def load_all_csvs_from_folder(folder_path):
    print(f"📥 Loading CSV files from folder: {folder_path}")

    files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".csv")
    )

    if not files:
        raise ValueError("❌ No CSV files found in folder.")

    print(f"📄 Found {len(files)} CSV files. Combining...")

    dfs = []
    for f in files:
        full_path = os.path.join(folder_path, f)
        print(f"   • Loading {f}")
        df = pd.read_csv(full_path)
        df["source_file"] = f   # Keep traceability
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"✅ Combined dataframe shape: {combined_df.shape}")

    return combined_df

def process_csv_folder(input_folder, output_csv):
    print("\n========================================")
    print(" 🗳 Processing Entire Booth (Multiple CSVs)")
    print("========================================\n")

    # Load + combine all CSVs
    df = load_all_csvs_from_folder(input_folder)

    # Normalize & prepare helper columns
    print("🔧 Normalizing fields...")
    df["serial_num"] = df["serial"].apply(to_int)
    df["age_num"] = df["age"].apply(to_int)

    df["name_norm"] = df["name"].apply(norm_str)
    df["house_norm"] = df["house_number"].apply(norm_str)
    df["id_clean"] = df["id"].apply(lambda x: norm_str(x).replace(" ", ""))

    df["father_norm"] = df["father_name"].apply(norm_str) if "father_name" in df.columns else ""
    df["husband_norm"] = df["husband_name"].apply(norm_str) if "husband_name" in df.columns else ""
    df["mother_norm"] = df["mother_name"].apply(norm_str) if "mother_name" in df.columns else ""
    df["gender_norm"] = df["gender"].apply(norm_str) if "gender" in df.columns else ""

    # Ensure numeric types for assembly/part
    if "assembly_number" in df.columns:
        df["assembly_number"] = df["assembly_number"].apply(to_int)
    else:
        df["assembly_number"] = math.nan

    if "part_no" in df.columns:
        df["part_no"] = df["part_no"].apply(to_int)
    else:
        df["part_no"] = 0

    # Initialize suspicious flags
    df["suspicious"] = False
    df["reasons"] = [[] for _ in range(len(df))]

    # Run all rules
    # run_rule_1_same_name_house_age_spread(df)
    # run_rule_2_duplicate_or_missing_id(df)
    # run_rule_3_duplicate_full_profile(df)
    run_rule_4_sequential_ids_in_house(df)
    run_rule_5_adults_missing_relations(df)
    run_rule_6_many_youth_in_house(df)
    run_rule_7_generic_name_no_relations(df)
    # run_rule_8_age_repeated_in_part(df)
    run_rule_9_serial_anomalies(df)
    run_rule_10_orphan_voter(df)
    run_rule_11_large_households(df, threshold=10)    

    # Turn reasons list into string
    df["reasons"] = df["reasons"].apply(lambda lst: "; ".join(lst))

    total_flagged = df["suspicious"].sum()
    print(f"\n========================================")
    print(f"✅ Finished all rules. Total flagged voters: {total_flagged}")
    print(f"💾 Writing merged flagged CSV to: {output_csv}")
    print("========================================\n")

    print(f"💾 Writing flagged CSV to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print("✅ Done.")

def main():
    parser = argparse.ArgumentParser(description="Flag suspicious voters from multiple CSV files in a folder.")
    parser.add_argument("--input_dir", required=True, help="Folder containing multiple CSV files for one booth.")
    parser.add_argument("--output", default="flagged_booth.csv", help="Output CSV file.")
    args = parser.parse_args()

    # Record the start time
    start_time = time.perf_counter() # Or time.time() for less precision

    process_csv_folder(args.input_dir, args.output)

    # Record the end time
    end_time = time.perf_counter() # Or time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Total time taken: {elapsed_time:.3f} seconds.")

if __name__ == "__main__":
    main()
