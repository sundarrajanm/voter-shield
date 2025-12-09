import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="VoterShield Viewer", layout="wide")

st.title("🛡️ VoterShield — Basic View")

FOLDER = "./page-csv"

@st.cache_data
def load_all_csvs(folder):
    all_rows = []

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            df = pd.read_csv(path)

            # Tag with source CSV (optional, useful for tracing)
            df["source_file"] = file  

            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    return combined

df = load_all_csvs(FOLDER)

st.write(f"**Total voters loaded: {len(df)}**")

st.subheader("Search Filters")

name_filter = st.text_input("Search Name")
epic_filter = st.text_input("Search EPIC ID")
house_filter = st.text_input("Search House Number")

filtered_df = df.copy()

def highlight_missing(val):
    if pd.isna(val) or val == "" or val == "null":
        return "background-color: #ffcccc"
    return ""

if name_filter:
    filtered_df = filtered_df[filtered_df["name"].str.contains(name_filter, case=False, na=False)]

if epic_filter:
    filtered_df = filtered_df[filtered_df["epic_id"].str.contains(epic_filter, case=False, na=False)]

if house_filter:
    filtered_df = filtered_df[filtered_df["house_no"].astype(str).str.contains(house_filter, case=False, na=False)]

# Render table
st.dataframe(
    filtered_df.style.applymap(highlight_missing),
    use_container_width=True
)

st.metric("Total Voters", len(df))
st.metric("Male", len(df[df["gender"] == "Male"]))
st.metric("Female", len(df[df["gender"] == "Female"]))
