import torch 
from torch.utils.data import IterableDataset
import polars as pl
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import get_worker_info
import math
from functools import cached_property

LEVELS = 20
OFI_LEVELS = 10

LOB_COLS = (
    [f"bid_p{i}" for i in range(LEVELS)] +
    [f"bid_q{i}" for i in range(LEVELS)] +
    [f"ask_p{i}" for i in range(LEVELS)] +
    [f"ask_q{i}" for i in range(LEVELS)]
)
GLOBAL_COLS = [f"ofi_{i}" for i in range(OFI_LEVELS)] + ['mid', 'spread']

SEQ_LEN = 100
HORIZON = 100
CHUNK_SIZE = 300_000
NN_BATCH_SIZE = 1024
MID_PRICE_INDEX = GLOBAL_COLS.index('mid')


class LOBDataset(IterableDataset):
    """
        Iterates through a parquet file in chunks of size chunk_size
        and yields pre-made batches of size nn_batch_size for 
        a neural network. 

        Crucially: this IterableDataset collates batches manually,
        the wrapping DataLoader must be called with batch_size=None.

        The logic of chunk processing: 
            - Create a polars df from the chunk,
            - Extract LOB features (prices/quantities),
            - Extract global features (ofi, mid, spread...),
            - Calculate target return bp from current mid price
              and mid price in some time horizon,
            - Manually collate LOB features to a tensor (Batch, Channels, Time, Levels),
              where Channels are [AskP, AskQ, BidP, BidQ],
            - Manually collate global features to a tensor (Batch, Time, Levels),
            - Yield LOB features, global features and targets as pytorch tensors.
        
        file_path: path to the parquet file with LOB data.
        horizon: time horizon for future mid price identification.
        seq_len: sequence length to be fed to CNN, the last element 
                 of the slice [i, ..., i + seq_len - 1] is treated as
                 current mid price.
        chunk_size: size of file chunks to iterate on.
        nn_batch_size: size of batches for the neural network.
    """

    def __init__(
            self, 
            split_dir: Path, 
            seq_len=SEQ_LEN, 
            horizon=HORIZON, 
            chunk_size=CHUNK_SIZE,
            nn_batch_size=NN_BATCH_SIZE
        ):
        self.split_dir = split_dir
        self.seq_len = seq_len
        self.horizon = horizon
        self.chunk_size = chunk_size
        self.nn_batch_size = nn_batch_size  

        self.req_len = self.seq_len + self.horizon
        self.shard_files = sorted(self.split_dir.glob('*.parquet'))
        print(self.shard_files)
        self.global_cols_mask = list(range(len(GLOBAL_COLS))) 
        self.global_cols_mask.pop(MID_PRICE_INDEX)  # Exclude mid price

    @staticmethod
    def _stack_lob(lob_np):
        bid_p = lob_np[:, 0 : LEVELS]
        bid_q = lob_np[:, LEVELS : 2*LEVELS]
        ask_p = lob_np[:, 2*LEVELS : 3*LEVELS]
        ask_q = lob_np[:, 3*LEVELS : 4*LEVELS]
        return np.stack([ask_p, ask_q, bid_p, bid_q], axis=-1)
    
    @cached_property
    def _worker_files(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self.shard_files
        else:
            per_worker = int(math.ceil(
                len(self.shard_files) / float(worker_info.num_workers)
            ))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.shard_files))
            return self.shard_files[start_idx : end_idx]

    def __iter__(self):
        print(self._worker_files)
        for file_path in self._worker_files:
            self.pf = pq.ParquetFile(file_path)
            overlap_lob = None
            overlap_global = None

            for chunk in self.pf.iter_batches(batch_size=self.chunk_size):
                df = pl.from_arrow(chunk)

                # Result dim: (Time, Channels * Levels)
                lob_np = df.select(LOB_COLS).to_numpy().astype(np.float32)

                # Result dim: (Time, Levels, Channels)
                lob_stack = self._stack_lob(lob_np)

                # Result dim: (Time, Levels)
                global_np = df.select(GLOBAL_COLS).to_numpy().astype(np.float32)
                
                if overlap_lob is not None:
                    lob_stack = np.concatenate([overlap_lob, lob_stack], axis=0)
                    global_np = np.concatenate([overlap_global, global_np], axis=0)
                
                n_rows = len(lob_stack)
                if n_rows < self.req_len:
                    overlap_lob = lob_stack
                    overlap_global = global_np
                    continue 
                
                valid_n = n_rows - self.req_len + 1

                mid_now_idx = np.arange(self.seq_len - 1, self.seq_len - 1 + valid_n)
                mid_future_idx = mid_now_idx + self.horizon
                mid_now = global_np[mid_now_idx, MID_PRICE_INDEX]
                mid_future = global_np[mid_future_idx, MID_PRICE_INDEX]
                target_bps = ((mid_future - mid_now) / (mid_now + 1e-8)) * 10_000

                # Result dim: (Batch, Levels, Channels, Time)
                lob_windows = sliding_window_view(
                    lob_stack[:valid_n + self.seq_len - 1],
                    window_shape=self.seq_len,
                    axis=0
                )
                # Result dim: (Batch, Channels, Time, Levels)
                # This is proper for CNN-GRU logic
                lob_windows = np.transpose(lob_windows, (0, 2, 3, 1))
                
                # Result dim: (Batch, OFI_LEVELS + 2, Time)
                global_windows = sliding_window_view(
                    global_np[:valid_n + self.seq_len - 1],
                    window_shape=self.seq_len,
                    axis=0
                )
                # Result dim: (Batch, Time, OFI_LEVELS + 2)
                global_windows = np.transpose(global_windows, (0, 2, 1))

                for i in range(0, valid_n, self.nn_batch_size):
                    end_i = min(i + self.nn_batch_size, valid_n)
                    batch_lob =  torch.from_numpy(np.ascontiguousarray(lob_windows[i : end_i]))
                    batch_global = torch.from_numpy(
                        np.ascontiguousarray(
                            # Result dim: (Batch, Time, OFI_LEVELS + 1)
                            # This is proper for CNN-GRU logic
                            global_windows[i : end_i, :, self.global_cols_mask]
                        )
                    )
                    batch_target = torch.from_numpy(target_bps[i : end_i]).unsqueeze(-1)
                    yield batch_lob, batch_global, batch_target
                
                overlap_lob = lob_stack[-(self.req_len - 1):].copy()    # Do not store an entire prev chunk in memory
                overlap_global = global_np[-(self.req_len - 1):].copy() # only copy the overlap

def get_dataset(date: str, type: str) -> LOBDataset | None:
    match type:
        case 'train':
            return LOBDataset(
                Path(f"data/splits/{date}/train"),
            )
        case 'val':
            return LOBDataset(
                Path(f"data/splits/{date}/val"),
            )
        case 'test':
            return LOBDataset(
                Path(f"data/splits/{date}/test"),
            )
        case _:
            return None
