import pandas as pd

df = pd.read_csv("data/code15/exams.csv")
# Keep only exams stored in exams_part0.hdf5
df_part0 = df[df["trace_file"] == "exams_part0.hdf5"].copy()
df_part0.to_csv("data/code15/exams_part0.csv", index=False)
print(f"Saved filtered CSV with {len(df_part0)} rows")
