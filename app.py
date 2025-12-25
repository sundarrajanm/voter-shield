import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="VoterShield Viewer", layout="wide")

st.title("🛡️ VoterShield — Basic View")

FOLDER = "./csv"


@st.cache_data
def load_all_csvs(folder):
    all_rows = []

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            df = pd.read_csv(path)

            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    return combined


df = load_all_csvs(FOLDER)

st.write(f"**Total voters loaded: {len(df)}**")

# ------------------------------------------------------------------
# 🔧 DATA FIXES FOR UI
# ------------------------------------------------------------------

# 1️⃣ Age as integer
# if "age" in df.columns:
#     df["age"] = df["age"].round(0).astype("Int64")

# ------------------------------------------------------

# st.subheader("Search Filters")

# show_only_suspicious = st.checkbox("Show suspicious voters")

# ------------------------------------------------------
# Rule Filter Dropdown (NEW)
# ------------------------------------------------------
# Extract unique rule names safely
# rule_options = (
#     df["rule"]
#         .fillna("")
#         .str.split(";")
#         .explode()
#         .str.strip()
#         .replace("", np.nan)
#         .dropna()
#         .unique()
# )

# rule_filter = st.selectbox(
#     "Filter by Rule",
#     options=["All Rules"] + sorted(rule_options)
# )

# cluster_id_filter = st.text_input("Family Cluster Id")
# section_filter = st.text_input("Section Number")
# name_filter = st.text_input("Name")
# epic_filter = st.text_input("EPIC ID")
# house_filter = st.text_input("House Number")

filtered_df = df.copy()

# if name_filter:
#     filtered_df = filtered_df[filtered_df["name"].str.contains(name_filter, case=False, na=False)]

# if epic_filter:
#     filtered_df = filtered_df[filtered_df["epic_id"].astype(str) == str(epic_filter)]

# if house_filter:
#     filtered_df = filtered_df[filtered_df["house_no"].astype(str) == str(house_filter)]

# if cluster_id_filter:
#     filtered_df = filtered_df[filtered_df["cluster_id"].astype(str) == str(cluster_id_filter)]

# if section_filter:
#     filtered_df = filtered_df[filtered_df["section_no"].astype(str) == str(section_filter)]

# if show_only_suspicious:
#     filtered_df = filtered_df[filtered_df["suspicious"] == True]

# Apply rule filter
# if rule_filter != "All Rules":
#     filtered_df = filtered_df[filtered_df["rule"].str.contains(rule_filter, na=False)]

# ------------------------------------------------------------------
# 🎨 STYLING
# ------------------------------------------------------------------


def highlight_missing(val):
    if pd.isna(val) or val == "" or val == "null":
        return "background-color: #ECECEC"
    return ""


def highlight_suspicious(row):
    if row["suspicious"]:
        return ["background-color: #FFE5B4"] * len(row)  # light peach/orange
    return [""] * len(row)


# Render table
# styled = (
#     filtered_df
#         .style
#         .apply(highlight_suspicious, axis=1) # row-level styling
#         .applymap(highlight_missing)         # cell-level styling
# )

# st.dataframe(styled, use_container_width=True)
# ------------------------------------------------------------------
# 🧾 TABLE RENDERING (IMPORTANT PART)
# ------------------------------------------------------------------

st.subheader("Voter Data")

st.data_editor(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "age": st.column_config.NumberColumn("Age", format="%d"),
        # "reasons": st.column_config.TextColumn(
        #     "Reasons",
        #     width="large"
        # ),
        # "rule": st.column_config.TextColumn(
        #     "Rule",
        #     width="large"
        # ),
        # "suspicious": st.column_config.CheckboxColumn(
        #     "Suspicious"
        # ),
    },
)

st.metric("Total Voters", len(df))
# st.metric("Suspicious Voters", len(filtered_df[filtered_df["suspicious"] == True]))
# st.metric("Unique Families", filtered_df["cluster_id"].nunique())
