import pyarrow.parquet as pq
from pathlib import Path
import polars as pl
import numpy as np
from numpy.typing import NDArray

LEVELS = 20
OFI_LEVELS = 10
HORIZON = 100
BATCH_SIZE = 300_000

COLS = (
    [f"bid_p{i}" for i in range(LEVELS)] +
    [f"bid_q{i}" for i in range(LEVELS)] +
    [f"ask_p{i}" for i in range(LEVELS)] +
    [f"ask_q{i}" for i in range(LEVELS)] + 
    [f"ofi_{i}" for i in range(OFI_LEVELS)] + 
    ['mid', 'spread']
)

MID_PRICE_INDEX = -2

def tabularize(file_path: Path, horizon: int = HORIZON, batch_size: int = BATCH_SIZE) -> tuple[NDArray, NDArray]:
    """
    Tabularize Limit Order Book.

    file_path: path to the parquet file.
    horizon: target horizon.
    batch_size: size of dataset chunks.
    """
    pf = pq.ParquetFile(file_path)

    X_list = []
    y_list = []

    overlap_features = None
    overlap_mid = None
    for batch in pf.iter_batches(batch_size=batch_size):
        df = pl.from_arrow(batch)
        features_np = df.select(COLS).to_numpy().astype(np.float32)
        mid_np = features_np[:, MID_PRICE_INDEX]

        if overlap_features is not None:
            future_mids = mid_np[:horizon] 
            future_len = len(future_mids)   # account for when len(future_mids) < horizon 
            curr_mid = overlap_mid[:future_len] # take only available current mid prices
            curr_features = overlap_features[:future_len] # and features
            raw_return = future_mids - curr_mid
            y_bps_overlap = (raw_return / curr_mid) * 10_000
            X_list.append(curr_features)
            y_list.append(y_bps_overlap)
        
        X_now = features_np[:-horizon]
        mid_now = mid_np[:-horizon]
        mid_future = mid_np[horizon:]
        y_bps = ((mid_future - mid_now) / mid_now) * 10_000
        X_list.append(X_now)
        y_list.append(y_bps)

        overlap_features = features_np[-horizon:].copy()
        overlap_mid = mid_np[-horizon:].copy()
    
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)