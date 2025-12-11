import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="VoterShield Viewer", layout="wide")

st.title("🛡️ VoterShield — Basic View")

FOLDER = "./final"

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

st.subheader("Search Filters")

show_only_suspicious = st.checkbox("Show suspicious voters")
cluster_id_filter = st.text_input("Family Cluster Id")
# section_filter = st.text_input("Section Number")
name_filter = st.text_input("Name")
epic_filter = st.text_input("EPIC ID")
house_filter = st.text_input("House Number")

filtered_df = df.copy()

def highlight_missing(val):
    if pd.isna(val) or val == "" or val == "null":
        return "background-color: #ECECEC"
    return ""

def highlight_suspicious(row):
    if row["suspicious"]:
        return ['background-color: #FFE5B4'] * len(row)   # light peach/orange
    return [''] * len(row)

if name_filter:
    filtered_df = filtered_df[filtered_df["name"].str.contains(name_filter, case=False, na=False)]

if epic_filter:
    filtered_df = filtered_df[filtered_df["epic_id"].str.contains(epic_filter, case=False, na=False)]

if house_filter:
    filtered_df = filtered_df[filtered_df["house_no"].astype(str).str.contains(house_filter, case=False, na=False)]

if cluster_id_filter:
    filtered_df = filtered_df[filtered_df["cluster_id"].astype(str) == str(cluster_id_filter)]

# if section_filter:
#     filtered_df = filtered_df[filtered_df["section_no"].astype(str) == str(section_filter)]

if show_only_suspicious:
    filtered_df = filtered_df[filtered_df["suspicious"] == True]

# Render table
styled = (
    filtered_df
        .style
        .apply(highlight_suspicious, axis=1) # row-level styling
        .applymap(highlight_missing)         # cell-level styling
)

st.dataframe(styled, use_container_width=True)

st.metric("Total Voters", len(filtered_df))
st.metric("Suspicious Voters", len(filtered_df[filtered_df["suspicious"] == True]))
st.metric("Unique Families", filtered_df["cluster_id"].nunique())
