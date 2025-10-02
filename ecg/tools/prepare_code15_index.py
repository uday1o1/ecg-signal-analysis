import argparse, os, json, h5py, pandas as pd, numpy as np
from tqdm import tqdm

ABN = ["1dAVb","RBBB","LBBB","SB","AF","ST"]

def build_examid_to_row(h5_path):
    with h5py.File(h5_path, "r") as f:
        exam_ids = np.array(f["exam_id"])  # name per dataset spec
    return {int(e): i for i, e in enumerate(exam_ids)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exams_csv", required=True, help="Path to exams.csv")
    ap.add_argument("--h5_dir", required=True, help="Directory with exams_part*.hdf5")
    ap.add_argument("--out_index_csv", default="./data/code15_index.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.exams_csv)
    # Keep rows that have a shard recorded
    assert "trace_file" in df.columns and "exam_id" in df.columns
    missing = df["trace_file"].isna()
    if missing.any():
        df = df[~missing].copy()

    # Build per-shard lookup: exam_id -> row
    shards = sorted({os.path.join(args.h5_dir, tf) for tf in df["trace_file"].unique()})
    id_maps = {}
    for shp in tqdm(shards, desc="Indexing shards"):
        id_maps[shp] = build_examid_to_row(shp)

    # Build index rows
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Building index"):
        exam_id = int(r["exam_id"])
        shard = os.path.join(args.h5_dir, r["trace_file"])
        row_in_shard = id_maps[shard].get(exam_id, None)
        if row_in_shard is None:
            continue
        label_vals = [int(r[a]) for a in ABN]  # assumes 0/1 already in CSV
        rows.append({
            "exam_id": exam_id,
            "shard": shard,
            "row": row_in_shard,
            **{f"y_{a}": v for a, v in zip(ABN, label_vals)}
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_index_csv) or ".", exist_ok=True)
    out.to_csv(args.out_index_csv, index=False)
    print(f"Wrote index: {args.out_index_csv} with {len(out)} rows")

if __name__ == "__main__":
    main()
