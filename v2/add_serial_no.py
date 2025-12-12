import pandas as pd

def add_serial_no(
    input_csv: str,
    output_csv: str = "voter_list_with_serial.csv",
    start_serial: int = 1
):
    """
    Reads voter_list.csv, sorts by source_file,
    assigns incremental serial_no, and writes output CSV.
    """

    print(f"📥 Reading {input_csv}")
    df = pd.read_csv(input_csv)

    # if "source_file" not in df.columns:
    #     raise ValueError("source_file column not found in CSV")
    #
    # # 1️⃣ Sort by source_file (and optionally by row order inside file)
    # df = df.sort_values(by=["source_file"]).reset_index(drop=True)

    # 2️⃣ Assign serial_no
    df["serial_no"] = range(start_serial, start_serial + len(df))

    # 3️⃣ Reorder columns: serial_no first, epic_id second
    cols = df.columns.tolist()
    cols.remove("serial_no")
    cols.remove("epic_id")

    new_order = ["serial_no", "epic_id"] + cols
    df = df[new_order]
    
    # 3️⃣ Save output
    df.to_csv(output_csv, index=False)

    print(f"✅ Serial numbers assigned")
    print(f"💾 Output written to {output_csv}")
    print(f"🔢 Serial range: {df['serial_no'].min()} → {df['serial_no'].max()}")

    return df


if __name__ == "__main__":
    add_serial_no(
        input_csv="voter_list.csv",
        output_csv="voter_list_with_serial.csv"
    )
