from typing import Iterator
import pyarrow.parquet as pq
import polars as pl
from pathlib import Path

def iter_batches(file_path: Path, batch_size: int) -> Iterator[pl.DataFrame]:
    """Yield batches from parquet file."""
    f = pq.ParquetFile(file_path)
    for batch in f.iter_batches(batch_size=batch_size):
        yield pl.from_arrow(batch)