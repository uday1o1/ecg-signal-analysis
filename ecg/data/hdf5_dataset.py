import h5py, numpy as np, torch
from torch.utils.data import Dataset

class HDF5ECG(Dataset):
    """
    Streams ECGs from an HDF5 dataset of shape [N, 4096, 12].
    If labels_arr is provided (np.ndarray [N, C]), returns (x, y); else only x.
    """
    def __init__(self, h5_path, dataset_name="tracings", labels_arr=None,
                 start_idx=0, end_idx=None, scale_by_1000=True, dtype=np.float32):
        self.f = h5py.File(h5_path, "r")
        self.x = self.f[dataset_name]
        self.labels = labels_arr
        self.start = start_idx
        self.end = end_idx if end_idx is not None else len(self.x)
        self.scale = 1000.0 if scale_by_1000 else 1.0
        self.dtype = dtype

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, i):
        j = self.start + i
        arr = np.array(self.x[j], dtype=self.dtype) * self.scale  # [4096,12]
        arr = np.transpose(arr, (1,0))  # [12,4096]
        x = torch.from_numpy(arr)
        if self.labels is None:
            return x
        y = torch.from_numpy(self.labels[j].astype(np.float32))
        return x, y

    def close(self):
        try: self.f.close()
        except: pass

    def __del__(self):
        self.close()
