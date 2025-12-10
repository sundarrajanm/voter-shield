import os
import pandas as pd

import os
import pandas as pd
import numpy as np

def augment_csv(
        df,
        part_no,
        constituency,
        section_no=None,
        section_name=None
    ):
    """
    Injects metadata columns into the dataframe.
    """

    df["part_no"] = part_no
    df["constituency"] = constituency

    if section_no is not None:
        df["section_no"] = section_no
    
    if section_name is not None:
        df["section_name"] = section_name

    return df


def drop_fully_empty_rows(df):
    """
    Removes rows where *all main voter fields* are empty/None/NaN.
    Metadata fields (part_no, constituency, section_no, section_name)
    are NOT considered when checking emptiness.
    """

    voter_fields = [
        "epic_id",
        "name",
        "father_name",
        "mother_name",
        "husband_name",
        "house_no",
        "age",
        "gender"
    ]

    # Normalize whitespace → empty
    df[voter_fields] = df[voter_fields].replace(
        {
            None: np.nan,
            "": np.nan,
            " ": np.nan,
            "  ": np.nan
        }
    )

    # Drop rows where all voter fields are NaN
    df = df.dropna(subset=voter_fields, how="all")

    return df


def process_csv_folder(
        folder_path,
        part_no,
        constituency,
        output_folder="augmented_csvs"
    ):
    """
    Reads all CSVs in the folder, drops empty rows,
    augments them, and writes output.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            input_path = os.path.join(folder_path, file)
            print(f"Processing {input_path} ...")

            df = pd.read_csv(input_path)

            # 1️⃣ Remove completely empty/noisy rows
            df = drop_fully_empty_rows(df)

            # Extract page number: page_03.csv → 3
            page_num = int(file.replace("page_", "").replace(".csv", ""))

            # Section-based assignment logic
            if page_num <= 18:
                section_no = 1
                section_name = "Karupparayan Kovil Street Ward No-9"
            else:
                section_no = 2
                section_name = "Water Tank Street Ward No-9"

            # 2️⃣ Metadata augmentation
            df = augment_csv(
                df,
                part_no=part_no,
                constituency=constituency,
                section_no=section_no,
                section_name=section_name
            )

            # 3️⃣ Save output
            output_path = os.path.join(output_folder, file)
            df.to_csv(output_path, index=False)

            print(f"Saved to {output_path}")

    print("\n🎉 All CSVs processed successfully!")


# RUN EXAMPLE
if __name__ == "__main__":

    FOLDER = "./page-csv"
    PART_NO = 244
    CONSTITUENCY = "116-SULUR"

    process_csv_folder(
        folder_path=FOLDER,
        part_no=PART_NO,
        constituency=CONSTITUENCY
    )
