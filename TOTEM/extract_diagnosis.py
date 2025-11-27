import pandas as pd
import numpy as np

def extract_last_alzheimers_status(
    input_csv="oasis3_clinical.csv",
    output_csv="patient_alzheimers_targets.csv"
):
    print(f"Reading {input_csv}...")
    # Load the big CSV (low_memory=False avoids warnings on mixed types)
    df = pd.read_csv(input_csv, low_memory=False)

    # 1. Identify the 'alzdis' column
    # The user mentioned it's column 1016, or named 'alzdis'. 
    # Let's try to find it dynamically to be safe.
    
    target_col = None
    candidates = [c for c in df.columns if "alzdis" in c.lower()]
    
    if candidates:
        target_col = candidates[0]
        print(f"Found target column by name: '{target_col}'")
    else:
        # Fallback to index 1016 if valid
        if len(df.columns) > 1016:
            target_col = df.columns[1016]
            print(f"Could not find 'alzdis' by name. Using column at index 1016: '{target_col}'")
        else:
            raise ValueError("Could not find a column named 'alzdis' and the file doesn't have 1016 columns.")

    # 1b. Pre-clean the target column
    # CRITICAL FIX: We convert "missing" strings to actual NaNs *before* grouping.
    # Pandas groupby().last() skips NaNs automatically.
    # This ensures that if the last visit is "missing", we look back to the previous valid visit.
    print(f"Pre-cleaning column '{target_col}' to handle 'missing' strings...")
    df[target_col] = df[target_col].replace(["missing", "nan", "NaN", "", " "], np.nan)

    # 2. Ensure Time Column exists for sorting
    # We need to know which value is the "last" one.
    if "days_to_visit" in df.columns:
        time_col = "days_to_visit"
        # Ensure it's numeric for sorting
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce').fillna(0)
    else:
        print("Warning: 'days_to_visit' not found. Using file order as time.")
        time_col = None

    # 3. Clean and Sort
    # Drop rows where the target is completely empty/missing/NaN immediately? 
    # Or keep them to see if the LAST one is valid?
    # Let's sort first.
    
    if time_col:
        df = df.sort_values(by=["patient_id", time_col])
    else:
        # Just sort by patient, preserving row order
        df = df.sort_values(by=["patient_id"])

    # 4. Group by Patient and take the LAST value
    print("Grouping by patient and extracting the last valid value...")
    
    # .last() returns the last NON-NULL value for each group.
    # Because we pre-cleaned "missing" -> NaN, this will effectively "fill forward" 
    # to find the most recent real diagnosis.
    last_records = df.groupby("patient_id").last().reset_index()
    
    # Select only ID and Target
    result_df = last_records[["patient_id", target_col]].copy()
    
    # Rename target column to a standard 'label' for easier downstream use
    result_df.rename(columns={target_col: "alzheimers_label"}, inplace=True)
    
    # 5. Final Cleanup
    # If a patient had NO valid values in their entire history, they will still be NaN here.
    initial_count = len(result_df)
    result_df = result_df.dropna(subset=["alzheimers_label"])
    final_count = len(result_df)
    
    print(f"Dropped {initial_count - final_count} patients with no valid label in their entire history.")
    
    # 6. Save
    result_df.to_csv(output_csv, index=False)
    print(f"Success! Saved {final_count} patient labels to {output_csv}")
    print(f"Sample:\n{result_df.head()}")

if __name__ == "__main__":
    extract_last_alzheimers_status()