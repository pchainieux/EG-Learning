from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Iterator
import math
import numpy as np

from src.data import mod_cog_tasks as mct
from neurogym import Dataset as NGDataset

@dataclass
class CachedDataset:
    X: np.ndarray
    Y: np.ndarray
    batch_first: bool = True 

    @property
    def num_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.X.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.X.shape[2])

    def sample_batch(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.num_samples, size=batch_size)
        return self.X[idx], self.Y[idx]

    def batches(self, batch_size: int, shuffle: bool = True, drop_last: bool = False,
                rng: Optional[np.random.Generator] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        N = self.num_samples
        idx = np.arange(N)
        if shuffle:
            rng = rng or np.random.default_rng()
            rng.shuffle(idx)
        n_chunks = N // batch_size if drop_last else math.ceil(N / batch_size)
        for k in range(n_chunks):
            sl = slice(k * batch_size, min((k + 1) * batch_size, N))
            take = idx[sl]
            yield self.X[take], self.Y[take]

class CachedBatcher:
    def __init__(self, cached_ds, batch_size, rng=None):
        import numpy as np
        self.ds = cached_ds
        self.bs = int(batch_size)
        self.rng = rng or np.random.default_rng()
    def __call__(self):
        return self.ds.sample_batch(self.bs, self.rng)

def build_cached_dataset(task_name: str,
                         num_batches: int,
                         batch_size: int,
                         seq_len: int,
                         *,
                         seed: Optional[int] = None,
                         dtype_float=np.float32,
                         dtype_int=np.int64) -> Tuple[CachedDataset, int]:
    if seed is not None:
        np.random.seed(seed)

    env_fn = getattr(mct, task_name)
    env = env_fn()

    ng_ds = NGDataset(env, batch_size=batch_size, seq_len=seq_len, batch_first=True)

    N = num_batches * batch_size
    X_list, Y_list = [], []
    produced = 0
    while produced < N:
        Xb, Yb = ng_ds()
        Xb = Xb.astype(dtype_float, copy=False)
        Yb = Yb.astype(dtype_int, copy=False)

        remaining = N - produced
        if Xb.shape[0] > remaining:
            Xb = Xb[:remaining]
            Yb = Yb[:remaining]

        X_list.append(Xb)
        Y_list.append(Yb)
        produced += Xb.shape[0]

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    assert X.shape[0] == N and Y.shape[0] == N

    cached = CachedDataset(X=X, Y=Y, batch_first=True)
    action_n = env.action_space.n
    return cached, action_n


def split_cached(ds: CachedDataset, val_frac: float = 0.1, *, seed: Optional[int] = None) -> Tuple[CachedDataset, CachedDataset]:
    N = ds.num_samples
    idx = np.arange(N)
    rng = np.random.default_rng(None if seed is None else seed)
    rng.shuffle(idx)

    n_val = int(round(val_frac * N))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr, Y_tr = ds.X[tr_idx], ds.Y[tr_idx]
    X_va, Y_va = ds.X[val_idx], ds.Y[val_idx]
    return CachedDataset(X_tr, Y_tr, ds.batch_first), CachedDataset(X_va, Y_va, ds.batch_first)


def save_cached_npz(path: str, ds: CachedDataset, *, meta: Optional[dict] = None) -> None:
    meta = meta or {}
    np.savez_compressed(path, X=ds.X, Y=ds.Y, meta=np.array([str(meta)], dtype=object))


def load_cached_npz(path: str) -> CachedDataset:
    with np.load(path, allow_pickle=True) as f:
        X = f["X"]
        Y = f["Y"]
    return CachedDataset(X=X, Y=Y, batch_first=True)
