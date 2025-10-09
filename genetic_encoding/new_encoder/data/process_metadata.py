import pandas as pd
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_file(vcf_path: Path) -> pd.DataFrame:
    """
    Read a single gzipped VCF and extract CHROM, POS, ID, REF, ALT, QUAL columns using pandas for speed.
    """
    df = pd.read_csv(
        vcf_path,
        sep='\t',
        comment='#',
        usecols=[0, 1, 2, 3, 4, 5],
        names=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL'],
        dtype={'CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str, 'QUAL': str},
        compression='gzip'
    )
    return df

def extract_parallel(input_dir: Path, output_csv: Path, workers: int):
    """
    Extract and aggregate SNP metadata from all matching VCF.gz files in parallel.
    """
    pattern = "ADNI.808_indiv.minGQ_21.pass.ADNI_ID.*.vcf.gz"
    vcf_paths = sorted(input_dir.glob(pattern), key=lambda p: p.name)

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        dfs = list(executor.map(process_file, vcf_paths))

    # Concatenate and sort
    combined = pd.concat(dfs, ignore_index=True)
    # Numeric sort of chromosomes where possible
    combined['CHROM_SORT'] = pd.to_numeric(combined['CHROM'], errors='coerce')
    combined.sort_values(['CHROM_SORT', 'CHROM', 'POS'], inplace=True)
    combined.drop(columns='CHROM_SORT', inplace=True)

    # Write to CSV
    combined.to_csv(output_csv, index=False)
    print(f"Extracted {len(combined)} variants into {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel SNP metadata extraction")
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Directory with ADNI.*.vcf.gz files")
    parser.add_argument("--output_csv", type=Path, default=Path("metadata.csv"),
                        help="Destination CSV file")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help="Number of parallel workers (default: number of CPU cores)")
    args = parser.parse_args()
    extract_parallel(args.input_dir, args.output_csv, args.workers)
