import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterator
import pyarrow as pa
import polars as pl

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
BATCH_SIZE  = 300_000
SEQ_LEN = 100
HORIZON = 100
OVERLAP_LEN = SEQ_LEN + HORIZON - 1
SHARD_SIZE = 500_000

def iter_batches(file_path: Path, batch_size: int) -> Iterator[pa.RecordBatch]:
    f = pq.ParquetFile(file_path)
    for batch in f.iter_batches(batch_size=batch_size):
        yield batch 

class ShardWriter:
    """
    Groups incoming dataframe slices into parquet shards.
    Prepends overlapping rows from the previous shard to
    the current shard.
    """
    def __init__(
            self, 
            output_dir: Path, 
            split_name: str, 
            schema: pa.Schema,
            shard_size: int,
            overlap_len: int
        ):
        if shard_size > overlap_len:
            raise ValueError('Shard size must be strictly greater than overlap length.')
        
        self.output_dir = output_dir / split_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split_name = split_name
        self.schema = schema
        self.buffer = pl.DataFrame()
        self.shard_idx = 0
        self.shard_size = shard_size
        self.overlap_len = overlap_len

    def _write(self, df: pl.DataFrame):
        out_path = self.output_dir / f"{self.split_name}_{self.shard_idx:04d}.parquet"
        pq.write_table(df.to_arrow(), out_path, compression='zstd')
        self.shard_idx += 1

    def write(self, df: pl.DataFrame):
        if df.is_empty():
            return
        
        if self.buffer.is_empty():
            self.buffer = df
        else:
            self.buffer = pl.concat([self.buffer, df]).rechunk()

        while len(self.buffer) >= self.shard_size:
            chunk = self.buffer.head(self.shard_size)
            self._write(chunk)
            next_start = self.shard_size - self.overlap_len
            self.buffer = self.buffer.slice(next_start)

    def close(self):
        if len(self.buffer) > self.overlap_len: # If buffer contains enough rows for prediction
            self._write(self.buffer)            # write it

def split(
        file_path: Path, 
        output_dir: Path, 
        train_ratio: float, 
        val_ratio: float, 
        shard_size: int, 
        overlap_len: int
    ):
    pf = pq.ParquetFile(file_path)
    n = pf.metadata.num_rows
    schema = pf.schema_arrow

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    print(f"Total rows: {n:,}")
    print(f"Train ends at: {train_end:,} & Val ends at: {val_end:,}")

    for split_name in ['train', 'val', 'test']:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    writers = {
        'train': ShardWriter(output_dir, 'train', schema, shard_size, overlap_len),
        'val': ShardWriter(output_dir, 'val', schema, shard_size, overlap_len),
        'test': ShardWriter(output_dir, 'test', schema, shard_size, overlap_len)
    }
    try:
        current_row = 0
        for batch in iter_batches(file_path, BATCH_SIZE):
            batch_len = len(batch)
            batch_start = current_row
            batch_end = current_row + batch_len

            df_batch = pl.from_arrow(batch)
            
            if batch_start < train_end:
                slice_end = min(batch_len, train_end - batch_start)
                train_df = df_batch.slice(0, slice_end)          
                writers['train'].write(train_df)

            if batch_end > train_end and batch_start < val_end:
                slice_start = max(0, train_end - batch_start)
                slice_end = min(batch_len, val_end - batch_start)
                val_df = df_batch.slice(slice_start, slice_end - slice_start)
                writers['val'].write(val_df)

            if batch_end > val_end:
                slice_start = max(0, val_end - batch_start)
                test_df = df_batch.slice(slice_start, batch_len - slice_start)
                writers['test'].write(test_df)

            current_row = batch_end
        print('\nSplit complete. Files sharded.')
    finally:
        for writer in writers.values():
            writer.close()

if __name__ == '__main__':
    input_file = Path("data/normalized/2026-04-26/lob20.parquet")
    output_dir = Path("data/splits/2026-04-26")
    
    split(
        input_file, 
        output_dir, 
        TRAIN_RATIO, 
        VAL_RATIO,
        SHARD_SIZE, 
        OVERLAP_LEN
    )