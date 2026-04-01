
import pandas as pd
import json
import os

CHUNK_SIZE = 1_000_000
OUTPUT_FILE = "hosp_patient_events.jsonl"

HOSP_FILES = {
    "services.csv.gz": {
        "kind": "service",
        "required": ["subject_id", "hadm_id", "transfertime", "curr_service"],
        "time_col": "transfertime",
        "usecols": ["subject_id", "hadm_id", "transfertime", "prev_service", "curr_service"],
    },
    "transfers.csv.gz": {
        "kind": "transfer",
        "required": ["subject_id", "hadm_id", "transfer_id", "intime", "eventtype"],
        "time_col": "intime",
        "usecols": ["subject_id", "hadm_id", "transfer_id", "eventtype", "careunit", "intime", "outtime"],
    },
}

VALID_SERVICE_RE = r"^[A-Z0-9]+$"
VALID_EVENTTYPE_RE = r"^[A-Za-z]+$"

def normalize_string(series):
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NONE": pd.NA})
    )

def process_services(chunk):
    chunk = chunk.copy()

    chunk["prev_service"] = normalize_string(chunk["prev_service"]).str.upper()
    chunk["curr_service"] = normalize_string(chunk["curr_service"]).str.upper()

    chunk["transfertime"] = pd.to_datetime(chunk["transfertime"], errors="coerce")

    chunk = chunk.dropna(subset=["subject_id", "hadm_id", "transfertime", "curr_service"])

    # curr_service is the main label for the service change; require it to be valid
    chunk = chunk[chunk["curr_service"].str.match(VALID_SERVICE_RE, na=False)].copy()

    # prev_service is optional; keep only if valid, otherwise null it out
    invalid_prev = ~chunk["prev_service"].str.match(VALID_SERVICE_RE, na=False)
    chunk.loc[invalid_prev.fillna(False), "prev_service"] = pd.NA

    chunk["event_time"] = chunk["transfertime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    chunk = chunk.drop_duplicates(
        subset=["subject_id", "hadm_id", "event_time", "prev_service", "curr_service"]
    ).sort_values(["subject_id", "hadm_id", "transfertime"])

    records = []
    for row in chunk.itertuples(index=False):
        records.append([
            int(row.subject_id),
            int(row.hadm_id),
            row.event_time,
            "service",
            {
                "curr_service": row.curr_service,
                "prev_service": None if pd.isna(row.prev_service) else row.prev_service,
            },
        ])
    return records

def process_transfers(chunk):
    chunk = chunk.copy()

    chunk["eventtype"] = normalize_string(chunk["eventtype"]).str.upper()
    chunk["careunit"] = normalize_string(chunk["careunit"])
    chunk["intime"] = pd.to_datetime(chunk["intime"], errors="coerce")
    chunk["outtime"] = pd.to_datetime(chunk["outtime"], errors="coerce")

    chunk = chunk.dropna(subset=["subject_id", "hadm_id", "transfer_id", "intime", "eventtype"])

    chunk = chunk[chunk["eventtype"].str.match(VALID_EVENTTYPE_RE, na=False)].copy()

    # Remove impossible intervals when both timestamps exist
    bad_interval = chunk["outtime"].notna() & (chunk["outtime"] < chunk["intime"])
    chunk = chunk[~bad_interval].copy()

    chunk["event_time"] = chunk["intime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    chunk["outtime_str"] = chunk["outtime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    chunk = chunk.drop_duplicates(
        subset=["subject_id", "hadm_id", "transfer_id", "eventtype", "careunit", "event_time", "outtime_str"]
    ).sort_values(["subject_id", "hadm_id", "intime", "transfer_id"])

    records = []
    for row in chunk.itertuples(index=False):
        records.append([
            int(row.subject_id),
            int(row.hadm_id),
            row.event_time,
            "transfer",
            {
                "transfer_id": int(row.transfer_id),
                "eventtype": row.eventtype,
                "careunit": None if pd.isna(row.careunit) else row.careunit,
                "outtime": None if pd.isna(row.outtime_str) else row.outtime_str,
            },
        ])
    return records

def process_and_append(file_name, config):
    if not os.path.exists(file_name):
        print(f"Skipping {file_name} - file not found.")
        return

    print(f"\nProcessing {file_name} in chunks...")
    processor = process_services if config["kind"] == "service" else process_transfers

    chunk_count = 0
    written = 0
    with open(OUTPUT_FILE, "a") as outfile:
        for chunk in pd.read_csv(file_name, usecols=config["usecols"], chunksize=CHUNK_SIZE, low_memory=False):
            chunk_count += 1
            records = processor(chunk)
            for record in records:
                outfile.write(json.dumps(record) + "\n")
            written += len(records)
            print(f"  -> chunk {chunk_count}: wrote {len(records):,} records", end="\r")

    print(f"\nFinished {file_name}. Total written: {written:,}")

def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    for file_name, config in HOSP_FILES.items():
        process_and_append(file_name, config)

    print(f"\nSuccess! Hospital events saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
