import h5py, numpy as np, torch, pandas as pd
from torch.utils.data import Dataset

class ShardedHDF5ECG(Dataset):
    """
    Reads samples from multiple HDF5 shards using an index CSV:
    columns: exam_id, shard, row, y_1dAVb, y_RBBB, y_LBBB, y_SB, y_AF, y_ST
    """
    ABN = ["1dAVb","RBBB","LBBB","SB","AF","ST"]

    def __init__(self, index_csv, scale_by_1000=True, dtype=np.float32, has_labels=True):
        self.df = pd.read_csv(index_csv).reset_index(drop=True)
        self.scale = 1000.0 if scale_by_1000 else 1.0
        self.dtype = dtype
        self.has_labels = has_labels and all((f"y_{a}" in self.df.columns) for a in self.ABN)
        self._open_files = {}

    def __len__(self): return len(self.df)

    def _h5(self, path):
        f = self._open_files.get(path)
        if f is None:
            f = h5py.File(path, "r")
            self._open_files[path] = f
        return f

    def __getitem__(self, i):
        row = self.df.iloc[i]
        f = self._h5(row["shard"])
        x = np.array(f["tracings"][int(row["row"])], dtype=self.dtype) * self.scale  # [4096, 12]
        x = np.transpose(x, (1,0))  # -> [12, 4096]
        xt = torch.from_numpy(x)
        if not self.has_labels:
            return xt
        y = torch.tensor([row[f"y_{a}"] for a in self.ABN], dtype=torch.float32)
        return xt, y

    def __del__(self):
        for f in self._open_files.values():
            try: f.close()
            except: pass
