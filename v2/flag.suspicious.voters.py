import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Rule 1: AGE_FLAGGED_WITHIN_A_CLUSTER
# ---------------------------------------------------------
def age_suspicion_in_cluster(group):
    """
    Detect age anomalies using actual family relationships:
    - Parent–child gap < 15 years
    - Husband–wife gap > 20 years
    """

    reasons = []

    # Create lookup: name → age
    name_to_age = (
        group[['name', 'age']]
        .dropna(subset=['name', 'age'])
        .set_index('name')['age']
        .to_dict()
    )

    # Iterate each voter inside the cluster
    for idx, row in group.iterrows():
        voter_name = row['name']
        voter_age = row['age']

        if pd.isna(voter_age):
            continue

        # -------------------------
        # A) Parent–Child Detection
        # -------------------------

        # Check father
        father = row['father_name']
        if pd.notna(father) and father in name_to_age:
            father_age = name_to_age[father]
            if abs(father_age - voter_age) < 15:
                reasons.append(
                    f"Parent–child age gap <15: {father} ({father_age}) & {voter_name} ({voter_age})"
                )

        # Check mother
        mother = row['mother_name']
        if pd.notna(mother) and mother in name_to_age:
            mother_age = name_to_age[mother]
            if abs(mother_age - voter_age) < 15:
                reasons.append(
                    f"Parent–child age gap <15: {mother} ({mother_age}) & {voter_name} ({voter_age})"
                )

        # -------------------------
        # B) Husband–Wife Detection
        # -------------------------

        husband = row['husband_name']
        if pd.notna(husband) and husband in name_to_age:
            husband_age = name_to_age[husband]
            if abs(husband_age - voter_age) > 20:
                reasons.append(
                    f"Husband–wife age gap >20: {husband} ({husband_age}) & {voter_name} ({voter_age})"
                )

    return reasons



# ---------------------------------------------------------
# Rule 2: MORE_THAN_10_VOTER_IN_SAME_HOUSE
# ---------------------------------------------------------
def house_overcrowding(group):
    count = len(group)
    if count > 10:
        return f"More than 10 voters in same house (count={count})"
    return None


# ---------------------------------------------------------
# Rule 3: ANOMALY_IN_DATA
# ---------------------------------------------------------
def anomaly_checks(row):
    reasons = []

    # --- Age anomaly ---
    if pd.notna(row['age']):
        if row['age'] < 18:
            reasons.append("Age < 18 (ineligible voter)")
        elif row['age'] > 90:
            reasons.append("Age > 90 (likely outdated record)")

    # # --- EPIC ID anomaly ---
    # epic = str(row['epic_id']).strip()
    # if len(epic) != 10:
    #     reasons.append(f"EPIC ID invalid length: '{epic}'")

    # --- Missing voter name ---
    if pd.isna(row['name']) or row['name'].strip() == "":
        reasons.append("Missing voter name")

    # --- Missing all family relationships ---

    has_father = pd.notna(row['father_name']) and str(row['father_name']).strip() != ""
    has_mother = pd.notna(row['mother_name']) and str(row['mother_name']).strip() != ""
    has_husband = pd.notna(row['husband_name']) and str(row['husband_name']).strip() != ""

    # If NONE of the relation fields exist → flag
    if not (has_father or has_mother or has_husband):
        reasons.append("Missing family relationship (no father, mother, or husband name)")

    return reasons



# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def process_voters(csv_path, output_path):
    df = pd.read_csv(csv_path)

    # Convert age to numbers safely
    df["age"] = pd.to_numeric(df["age"], errors="coerce")    

    # Initialize columns
    df['reasons'] = [[] for _ in range(len(df))]
    df['suspicious'] = False

    # # -----------------------------------------------------
    # # Apply Rule 1: Age anomaly within clusters
    # # -----------------------------------------------------
    # for cluster_id, group in df.groupby("cluster_id"):

    #     # Skip non-family clusters
    #     if cluster_id == -1:            
    #         continue

    #     reasons = age_suspicion_in_cluster(group)
    #     if reasons:
    #         df.loc[group.index, 'reasons'] = df.loc[group.index, 'reasons'].apply(lambda lst: lst + reasons)
    #         df.loc[group.index, 'suspicious'] = True

    # -----------------------------------------------------
    # Apply Rule 2: More than 10 voters in same house+street
    # Key: section_no + house_no
    # -----------------------------------------------------
    for (section, house), group in df.groupby(["section_no", "house_no"]):
        r = house_overcrowding(group)
        if r:
            df.loc[group.index, 'reasons'] = df.loc[group.index, 'reasons'].apply(lambda lst: lst + [r])
            df.loc[group.index, 'suspicious'] = True

    # -----------------------------------------------------
    # Apply Rule 3: Individual data anomalies
    # -----------------------------------------------------
    for i, row in df.iterrows():
        r = anomaly_checks(row)
        if r:
            df.at[i, 'reasons'] = df.at[i, 'reasons'] + r
            df.at[i, 'suspicious'] = True

    # Convert list → string
    df['reasons'] = df['reasons'].apply(lambda x: "; ".join(x) if x else "")

    # Save
    df.to_csv(output_path, index=False)
    print(f"Completed. Output saved to {output_path}")

    return df


# ---------------------------------------------------------
# Execute
# ---------------------------------------------------------
if __name__ == "__main__":
    process_voters("./voter_list.csv", "./final/voters_flagged.csv")
