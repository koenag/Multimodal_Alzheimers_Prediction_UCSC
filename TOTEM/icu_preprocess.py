import pandas as pd
import json
import os

# --- Configuration (ICU ONLY) ---
D_ITEMS_FILE = 'd_items.csv'
OUTPUT_FILE = 'icu_patient_events.jsonl'
CHUNK_SIZE = 1_000_000 # Process 1 million rows at a time

# Map out the exact column names for the ICU files based on your screenshots
EVENT_FILES = {
    'chartevents.csv': {'time_col': 'charttime', 'val_col': 'value', 'uom_col': 'valueuom'},
    'outputevents.csv': {'time_col': 'charttime', 'val_col': 'value', 'uom_col': 'valueuom'},
    'datetimeevents.csv': {'time_col': 'charttime', 'val_col': 'value', 'uom_col': 'valueuom'},
    'ingredientevents.csv': {'time_col': 'starttime', 'val_col': 'amount', 'uom_col': 'amountuom'},
    'procedureevents.csv': {'time_col': 'starttime', 'val_col': 'value', 'uom_col': 'valueuom'}
}

def load_dictionaries():
    print("Loading ICU dictionary mapping...")
    # Load Item dictionary
    d_items = pd.read_csv(D_ITEMS_FILE, usecols=['itemid', 'label'], low_memory=False)
    
    # Create mapping dictionary: {itemid: "Label Name"}
    items_map = dict(zip(d_items['itemid'], d_items['label']))
    return items_map

def process_and_append(file_name, config, items_map):
    if not os.path.exists(file_name):
        print(f"Skipping {file_name} - File not found in directory.")
        return

    print(f"\nProcessing {file_name} in chunks...")
    cols_to_use = ['subject_id', config['time_col'], 'itemid', config['val_col'], config['uom_col']]
    
    chunk_count = 0
    
    with open(OUTPUT_FILE, 'a') as outfile:
        # Read the CSV in manageable chunks
        for chunk in pd.read_csv(file_name, usecols=cols_to_use, chunksize=CHUNK_SIZE, low_memory=False):
            chunk_count += 1
            print(f"  -> Processing chunk {chunk_count} ({chunk_count * CHUNK_SIZE:,} rows)", end='\r')
            
            # Drop rows where the value is missing (no event recorded)
            chunk = chunk.dropna(subset=[config['val_col']])
            
            # Map the itemid to the human-readable Label
            chunk['parameter'] = chunk['itemid'].map(items_map).fillna('Unknown Parameter')
            
            # Combine the value and the unit of measurement (e.g., "80" + "bpm" -> "80 bpm")
            chunk['formatted_value'] = chunk[config['val_col']].astype(str) + " " + chunk[config['uom_col']].fillna('').astype(str)
            
            # Clean up extra spaces if there is no unit of measurement
            chunk['formatted_value'] = chunk['formatted_value'].str.strip()
            
            # Keep only the essential columns
            final_data = chunk[['subject_id', config['time_col'], 'parameter', 'formatted_value']]
            
            # Convert each row to the requested list format: [patient_id, date, parameter, value]
            records = final_data.values.tolist()
            
            for record in records:
                # Write each event as a JSON array line
                outfile.write(json.dumps(record) + '\n')

    print(f"\nFinished processing {file_name}.")

def main():
    # Clear previous output if it exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    items_map = load_dictionaries()
    
    for file_name, config in EVENT_FILES.items():
        process_and_append(file_name, config, items_map)
        
    print(f"\nSuccess! All ICU events have been saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()