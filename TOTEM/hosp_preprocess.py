import json
import os
import re
import pandas as pd

OUTPUT_EVENTS = "hosp_patient_events.jsonl"
CHUNK_SIZE = 500_000
DATA_DIR = r"C:\Users\kanei\Downloads"


def clean_str(s):
    return s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def parse_dt(s):
    return pd.to_datetime(s, errors="coerce")


def valid_provider_id(s):
    return clean_str(s).str.upper().where(lambda x: x.str.match(r"^P[0-9A-Z]+$", na=False))


def append_events(df, time_col, event_name, payload_builder, extra_sort=None):
    if df.empty:
        return 0
    sort_cols = ["subject_id", "hadm_id", time_col] + (extra_sort or [])
    df = df.drop_duplicates().sort_values(sort_cols)
    count = 0
    with open(OUTPUT_EVENTS, "a", encoding="utf-8") as out:
        for row in df.itertuples(index=False):
            out.write(json.dumps([
                int(row.subject_id),
                int(row.hadm_id),
                str(getattr(row, time_col)),
                event_name,
                payload_builder(row),
            ], default=str) + "\n")
            count += 1
    return count


def process_in_chunks(file_name, usecols, chunk_fn):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        print(f"Skipping {file_name} - not found.")
        return 0
    total = 0
    reader = pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False, compression="gzip")
    for i, chunk in enumerate(reader, 1):
        total += chunk_fn(chunk)
        print(f"Processed {file_name} chunk {i}", end="\r")
    print(f"Processed {file_name}: wrote {total:,} rows")
    return total


def process_services():
    valid = {
        "CMED", "CSURG", "DENT", "ENT", "GU", "GYN", "MED", "NB", "NBB",
        "NEURO", "NSURG", "OBS", "ORTHO", "OMED", "PSURG", "PSYCH", "SURG",
        "TRAUM", "TSURG", "VSURG"
    }

    def handle(chunk):
        chunk = chunk.dropna(subset=["subject_id", "hadm_id", "transfertime", "curr_service"]).copy()
        for c in ["prev_service", "curr_service"]:
            chunk.loc[:, c] = clean_str(chunk[c]).str.upper()
        chunk.loc[:, "transfertime"] = parse_dt(chunk["transfertime"])
        chunk = chunk.dropna(subset=["transfertime"]).copy()
        chunk = chunk.loc[chunk["curr_service"].isin(valid)].copy()
        chunk.loc[:, "prev_service"] = chunk["prev_service"].where(chunk["prev_service"].isin(valid))
        return append_events(
            chunk,
            "transfertime",
            "service",
            lambda r: {"curr_service": r.curr_service, "prev_service": None if pd.isna(r.prev_service) else r.prev_service},
        )

    return process_in_chunks(
        "services.csv.gz",
        ["subject_id", "hadm_id", "transfertime", "prev_service", "curr_service"],
        handle,
    )


def process_transfers():
    valid_eventtypes = {"ADMIT", "TRANSFER", "DISCHARGE", "ED", "DIRECT OBSERVATION", "EU OBSERVATION"}

    def handle(chunk):
        chunk = chunk.dropna(subset=["subject_id", "hadm_id", "transfer_id", "intime", "eventtype"]).copy()
        for c in ["eventtype", "careunit"]:
            chunk.loc[:, c] = clean_str(chunk[c]).str.upper()
        chunk.loc[:, "intime"] = parse_dt(chunk["intime"])
        chunk.loc[:, "outtime"] = parse_dt(chunk["outtime"])
        chunk = chunk.dropna(subset=["intime", "eventtype"]).copy()
        chunk = chunk.loc[chunk["eventtype"].isin(valid_eventtypes)].copy()
        chunk = chunk.loc[chunk["outtime"].isna() | (chunk["outtime"] >= chunk["intime"])].copy()
        return append_events(
            chunk,
            "intime",
            "transfer",
            lambda r: {
                "transfer_id": int(r.transfer_id),
                "eventtype": r.eventtype,
                "careunit": None if pd.isna(r.careunit) else r.careunit,
                "outtime": None if pd.isna(r.outtime) else str(r.outtime),
            },
        )

    return process_in_chunks(
        "transfers.csv.gz",
        ["subject_id", "hadm_id", "transfer_id", "eventtype", "careunit", "intime", "outtime"],
        handle,
    )


def process_procedures_icd():
    def handle(chunk):
        chunk = chunk.dropna(subset=["subject_id", "hadm_id", "chartdate", "seq_num", "icd_code", "icd_version"]).copy()
        chunk.loc[:, "chartdate"] = parse_dt(chunk["chartdate"])
        chunk.loc[:, "icd_code"] = clean_str(chunk["icd_code"]).str.upper()
        chunk = chunk.dropna(subset=["chartdate", "icd_code"]).copy()
        chunk = chunk.loc[chunk["icd_version"].isin([9, 10])].copy()
        chunk = chunk.loc[chunk["seq_num"] >= 1].copy()
        chunk = chunk.loc[chunk["icd_code"].str.match(r"^[A-Z0-9.]+$", na=False)].copy()
        return append_events(
            chunk,
            "chartdate",
            "procedure_icd",
            lambda r: {"seq_num": int(r.seq_num), "icd_code": r.icd_code, "icd_version": int(r.icd_version)},
            extra_sort=["seq_num"],
        )

    return process_in_chunks(
        "procedures_icd.csv.gz",
        ["subject_id", "hadm_id", "chartdate", "seq_num", "icd_code", "icd_version"],
        handle,
    )


def process_microbiology():
    usecols = [
        "subject_id", "hadm_id", "chartdate", "charttime", "spec_type_desc", "test_name",
        "org_name", "ab_name", "interpretation", "comments", "order_provider_id"
    ]

    def handle(chunk):
        chunk = chunk.dropna(subset=["subject_id", "hadm_id"]).copy()
        for c in ["spec_type_desc", "test_name", "org_name", "ab_name", "interpretation", "comments"]:
            chunk.loc[:, c] = clean_str(chunk[c])
        chunk.loc[:, "event_time"] = parse_dt(chunk["charttime"])
        missing = chunk["event_time"].isna()
        chunk.loc[missing, "event_time"] = parse_dt(chunk.loc[missing, "chartdate"])
        chunk = chunk.dropna(subset=["event_time"]).copy()
        informative = chunk[["spec_type_desc", "test_name", "org_name", "ab_name", "interpretation", "comments"]].notna().any(axis=1)
        chunk = chunk.loc[informative].copy()
        chunk.loc[:, "order_provider_id"] = valid_provider_id(chunk["order_provider_id"])
        return append_events(
            chunk,
            "event_time",
            "microbiology",
            lambda r: {k: (None if pd.isna(v) else v) for k, v in {
                "spec_type_desc": r.spec_type_desc,
                "test_name": r.test_name,
                "org_name": r.org_name,
                "ab_name": r.ab_name,
                "interpretation": r.interpretation,
                "comments": r.comments,
                "order_provider_id": r.order_provider_id,
            }.items() if not pd.isna(v)},
        )

    return process_in_chunks("microbiologyevents.csv.gz", usecols, handle)


def process_omr():
    def handle(chunk):
        chunk = chunk.dropna(subset=["subject_id", "chartdate", "result_name", "result_value"]).copy()
        chunk.loc[:, "chartdate"] = parse_dt(chunk["chartdate"])
        for c in ["result_name", "result_value"]:
            chunk.loc[:, c] = clean_str(chunk[c])
        chunk = chunk.dropna(subset=["chartdate", "result_name", "result_value"]).copy()
        chunk.loc[:, "result_name"] = chunk["result_name"].str.upper()
        chunk = chunk.loc[chunk["result_name"].str.len().between(2, 100)].copy()
        chunk = chunk.loc[chunk["result_value"].str.len().between(1, 100)].copy()
        chunk.loc[:, "hadm_id"] = -1
        return append_events(
            chunk,
            "chartdate",
            "omr",
            lambda r: {"result_name": r.result_name, "result_value": r.result_value},
        )

    return process_in_chunks("omr.csv.gz", ["subject_id", "chartdate", "result_name", "result_value"], handle)


def process_prescriptions():
    usecols = [
        "subject_id", "hadm_id", "starttime", "stoptime", "drug", "drug_type", "route",
        "dose_val_rx", "dose_unit_rx", "form_rx", "ndc", "gsn"
    ]

    def handle(chunk):
        chunk = chunk.dropna(subset=["subject_id", "hadm_id", "starttime", "drug"]).copy()
        for c in ["drug", "drug_type", "route", "dose_val_rx", "dose_unit_rx", "form_rx", "ndc", "gsn"]:
            chunk.loc[:, c] = clean_str(chunk[c])
        chunk.loc[:, "drug"] = chunk["drug"].str.upper()
        chunk.loc[:, "route"] = chunk["route"].str.upper()
        chunk.loc[:, "starttime"] = parse_dt(chunk["starttime"])
        chunk.loc[:, "stoptime"] = parse_dt(chunk["stoptime"])
        chunk = chunk.dropna(subset=["starttime", "drug"]).copy()
        chunk = chunk.loc[chunk["drug"].str.len() > 1].copy()
        chunk = chunk.loc[chunk["stoptime"].isna() | (chunk["stoptime"] >= chunk["starttime"])].copy()
        chunk.loc[:, "ndc"] = chunk["ndc"].where(chunk["ndc"].fillna("").str.match(r"^[0-9\-]+$", na=False))
        chunk.loc[:, "route"] = chunk["route"].where(chunk["route"].fillna("").str.len().between(1, 30))
        return append_events(
            chunk,
            "starttime",
            "prescription",
            lambda r: {k: v for k, v in {
                "drug": r.drug,
                "drug_type": None if pd.isna(r.drug_type) else r.drug_type,
                "route": None if pd.isna(r.route) else r.route,
                "dose_val_rx": None if pd.isna(r.dose_val_rx) else r.dose_val_rx,
                "dose_unit_rx": None if pd.isna(r.dose_unit_rx) else r.dose_unit_rx,
                "form_rx": None if pd.isna(r.form_rx) else r.form_rx,
                "ndc": None if pd.isna(r.ndc) else r.ndc,
                "gsn": None if pd.isna(r.gsn) else r.gsn,
                "stoptime": None if pd.isna(r.stoptime) else str(r.stoptime),
            }.items() if v is not None},
        )

    return process_in_chunks("prescriptions.csv.gz", usecols, handle)


def process_provider():
    f = os.path.join(DATA_DIR, "provider.csv.gz")
    if not os.path.exists(f):
        print("Skipping provider.csv.gz - not found.")
        return 0
    df = pd.read_csv(f, usecols=["provider_id"], low_memory=False, compression="gzip")
    df["provider_id"] = valid_provider_id(df["provider_id"])
    df = df.dropna(subset=["provider_id"]).drop_duplicates().sort_values(["provider_id"])
    df.to_csv("provider_preprocessed.csv", index=False)
    print(f"Processed provider.csv.gz: wrote {len(df):,} rows")
    return len(df)


def process_patients():
    f = os.path.join(DATA_DIR, "patients.csv.gz")
    if not os.path.exists(f):
        print("Skipping patients.csv.gz - not found.")
        return 0
    cols = ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"]
    df = pd.read_csv(f, usecols=cols, low_memory=False, compression="gzip")
    df = df.dropna(subset=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"])
    df["gender"] = clean_str(df["gender"]).str.upper()
    df["anchor_year_group"] = clean_str(df["anchor_year_group"])
    df["dod"] = parse_dt(df["dod"])
    df = df[df["gender"].isin(["M", "F"])]
    df = df[df["anchor_age"].between(0, 120)]
    df = df[df["anchor_year"].between(1900, 2200)]
    df = df.drop_duplicates(subset=["subject_id"]).sort_values(["subject_id"])
    df.to_csv("patients_preprocessed.csv", index=False)
    print(f"Processed patients.csv.gz: wrote {len(df):,} rows")
    return len(df)


def main():
    if os.path.exists(OUTPUT_EVENTS):
        os.remove(OUTPUT_EVENTS)
    process_services()
    process_transfers()
    process_procedures_icd()
    process_microbiology()
    process_omr()
    process_prescriptions()
    process_provider()
    process_patients()
    print(f"\nDone. Event timeline saved to {OUTPUT_EVENTS}")


if __name__ == "__main__":
    main()
