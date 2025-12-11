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
    rules = []

    name_to_age = (
        group[['name', 'age']]
        .dropna(subset=['name', 'age'])
        .set_index('name')['age']
        .to_dict()
    )

    for idx, row in group.iterrows():
        voter_name = row['name']
        voter_age = row['age']
        if pd.isna(voter_age):
            continue

        # Parent–child: father
        father = row['father_name']
        if pd.notna(father) and father in name_to_age:
            if abs(name_to_age[father] - voter_age) < 15:
                reasons.append(
                    f"Parent–child age gap <15: {father} ({name_to_age[father]}) & {voter_name} ({voter_age})"
                )
                rules.append("AGE_FLAGGED_WITHIN_A_CLUSTER")

        # Parent–child: mother
        mother = row['mother_name']
        if pd.notna(mother) and mother in name_to_age:
            if abs(name_to_age[mother] - voter_age) < 15:
                reasons.append(
                    f"Parent–child age gap <15: {mother} ({name_to_age[mother]}) & {voter_name} ({voter_age})"
                )
                rules.append("AGE_FLAGGED_WITHIN_A_CLUSTER")

        # Husband–Wife
        husband = row['husband_name']
        if pd.notna(husband) and husband in name_to_age:
            if abs(name_to_age[husband] - voter_age) > 20:
                reasons.append(
                    f"Husband–wife age gap >20: {husband} ({name_to_age[husband]}) & {voter_name} ({voter_age})"
                )
                rules.append("AGE_FLAGGED_WITHIN_A_CLUSTER")

    return reasons, rules



# ---------------------------------------------------------
# Rule 2: MORE_THAN_10_VOTER_IN_SAME_HOUSE
# ---------------------------------------------------------
def house_overcrowding(group):
    count = len(group)
    if count > 10:
        return (
            f"More than 10 voters in same house (count={count})",
            "MORE_THAN_10_VOTER_IN_SAME_HOUSE"
        )
    return None, None



# ---------------------------------------------------------
# Rule 3: ANOMALY_IN_DATA
# ---------------------------------------------------------
def anomaly_checks(row):
    reasons = []
    rules = []

    # Age anomaly
    if pd.notna(row['age']) and row['age'] < 18:
        reasons.append("Age < 18 (ineligible voter)")
        rules.append("ANOMALY_IN_DATA")

    # Missing voter name
    if pd.isna(row['name']) or row['name'].strip() == "":
        reasons.append("Missing voter name")
        rules.append("ANOMALY_IN_DATA")

    # Missing all family relationships
    has_father = pd.notna(row['father_name']) and row['father_name'].strip() != ""
    has_mother = pd.notna(row['mother_name']) and row['mother_name'].strip() != ""
    has_husband = pd.notna(row['husband_name']) and row['husband_name'].strip() != ""

    if not (has_father or has_mother or has_husband):
        reasons.append("Missing family relationship (no father, mother, or husband name)")
        rules.append("ANOMALY_IN_DATA")

    return reasons, rules



# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def process_voters(csv_path, output_path):
    df = pd.read_csv(csv_path)

    # Convert age
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Initialize output columns
    df['reasons'] = [[] for _ in range(len(df))]
    df['Rule'] = [[] for _ in range(len(df))]
    df['suspicious'] = False

    # # -----------------------------------------------------
    # # Apply Rule 1: Age anomaly within clusters
    # # -----------------------------------------------------
    # for cluster_id, group in df.groupby("cluster_id"):
    #     if cluster_id == -1:
    #         continue
    #
    #     reasons, rules = age_suspicion_in_cluster(group)
    #     if reasons:
    #         df.loc[group.index, 'reasons'] = df.loc[group.index, 'reasons'].apply(lambda lst: lst + reasons)
    #         df.loc[group.index, 'Rule'] = df.loc[group.index, 'Rule'].apply(lambda lst: lst + rules)
    #         df.loc[group.index, 'suspicious'] = True

    # -----------------------------------------------------
    # Apply Rule 2: House > 10 voters
    # -----------------------------------------------------
    for (section, house), group in df.groupby(["section_no", "house_no"]):
        reason, rule = house_overcrowding(group)
        if reason:
            df.loc[group.index, 'reasons'] = df.loc[group.index, 'reasons'].apply(lambda lst: lst + [reason])
            df.loc[group.index, 'Rule'] = df.loc[group.index, 'Rule'].apply(lambda lst: lst + [rule])
            df.loc[group.index, 'suspicious'] = True

    # -----------------------------------------------------
    # Apply Rule 3: Individual anomalies
    # -----------------------------------------------------
    for i, row in df.iterrows():
        reasons, rules = anomaly_checks(row)
        if reasons:
            df.at[i, 'reasons'] = df.at[i, 'reasons'] + reasons
            df.at[i, 'Rule'] = df.at[i, 'Rule'] + rules
            df.at[i, 'suspicious'] = True

    # Convert arrays → strings
    df['reasons'] = df['reasons'].apply(lambda x: "; ".join(x))
    df['Rule'] = df['Rule'].apply(lambda x: "; ".join(sorted(set(x))) if x else "")

    df.to_csv(output_path, index=False)
    print(f"Completed. Output saved to {output_path}")

    return df



# ---------------------------------------------------------
# Execute
# ---------------------------------------------------------
if __name__ == "__main__":
    process_voters("./voter_list.csv", "./final/voters_flagged.csv")
