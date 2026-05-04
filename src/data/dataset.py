import torch 
from torch.utils.data import IterableDataset
import polars as pl
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

LEVELS = 20
OFI_LEVELS = 10
SEQ_LEN = 100
HORIZON = 100
CHUNK_SIZE = 300_000
NN_BATCH_SIZE = 1024
MID_PRICE_INDEX = -2

LOB_COLS = (
    [f"bid_p{i}" for i in range(LEVELS)] +
    [f"bid_q{i}" for i in range(LEVELS)] +
    [f"ask_p{i}" for i in range(LEVELS)] +
    [f"ask_q{i}" for i in range(LEVELS)]
)
GLOBAL_COLS = [f"ofi_{i}" for i in range(OFI_LEVELS)] + ['mid', 'spread']

class LOBDataset(IterableDataset):
    """
        Iterates through a parquet file in chunks of size chunk_size
        and yields pre-made batches of size nn_batch_size for 
        a neural network. 

        Crucially: this IterableDataset collates batches manually,
        the wrapping DataLoader must be called with batch_size=None 
        and num_workers=0 (TODO: solve the latter).

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
            file_path, 
            seq_len=SEQ_LEN, 
            horizon=HORIZON, 
            chunk_size=CHUNK_SIZE,
            nn_batch_size=NN_BATCH_SIZE
        ):
        self.file_path = file_path
        self.seq_len = seq_len
        self.horizon = horizon
        self.chunk_size = chunk_size
        self.nn_batch_size = nn_batch_size

        self.pf = pq.ParquetFile(self.file_path)
        self.req_len = self.seq_len + self.horizon

    @staticmethod
    def _stack_lob(lob_np):
        bid_p = lob_np[:, 0 : LEVELS]
        bid_q = lob_np[:, LEVELS : 2*LEVELS]
        ask_p = lob_np[:, 2*LEVELS : 3*LEVELS]
        ask_q = lob_np[:, 3*LEVELS : 4*LEVELS]
        return np.stack([ask_p, ask_q, bid_p, bid_q], axis=-1)

    def __iter__(self):
        overlap_lob = None
        overlap_global = None
        for chunk in self.pf.iter_batches(batch_size=self.chunk_size):
            df = pl.from_arrow(chunk)

            # Result dim: (Time, Levels)
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
            
            # Result dim: (Batch, Levels, Time)
            global_windows = sliding_window_view(
                global_np[:valid_n + self.seq_len - 1],
                window_shape=self.seq_len,
                axis=0
            )
            # Result dim: (Batch, Time, Levels)
            # This is proper for CNN-GRU logic
            global_windows = np.transpose(global_windows, (0, 2, 1))

            for i in range(0, valid_n, self.nn_batch_size):
                end_i = min(i + self.nn_batch_size, valid_n)
                batch_lob =  torch.from_numpy(np.ascontiguousarray(lob_windows[i : end_i]))
                batch_global = torch.from_numpy(np.ascontiguousarray(global_windows[i : end_i]))
                batch_target = torch.from_numpy(target_bps[i : end_i]).unsqueeze(-1)
                yield batch_lob, batch_global, batch_target
            
            overlap_lob = lob_stack[-(self.req_len - 1):].copy()    # Do not store an entire prev chunk in memory
            overlap_global = global_np[-(self.req_len - 1):].copy() # only copy the overlap

def get_dataset(date: str, type: str) -> LOBDataset | None:
    match type:
        case 'train':
            return LOBDataset(
                Path(f"data/splits/{date}/train/train.parquet"),
            )
        case 'val':
            return LOBDataset(
                Path(f"data/splits/{date}/val/val.parquet"),
            )
        case 'test':
            return LOBDataset(
                Path(f"data/splits/{date}/test/test.parquet"),
            )
        case _:
            return None
