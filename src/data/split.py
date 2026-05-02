import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterator
import pyarrow as pa
import polars as pl

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
BATCH_SIZE  = 300_000

def iter_batches(file_path: Path, batch_size: int) -> Iterator[pa.RecordBatch]:
    f = pq.ParquetFile(file_path)
    for batch in f.iter_batches(batch_size=batch_size):
        yield batch 

def split(file_path: Path, output_dir: Path, train_ratio: float, val_ratio: float):
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
        'train': pq.ParquetWriter(output_dir / 'train' / 'train.parquet', schema, compression='zstd'),
        'val': pq.ParquetWriter(output_dir / 'val' / 'val.parquet', schema, compression='zstd'),
        'test': pq.ParquetWriter(output_dir / 'test' / 'test.parquet', schema, compression='zstd')
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
                if len(train_df) > 0:
                    writers['train'].write_table(train_df.to_arrow())

            if batch_end > train_end and batch_start < val_end:
                slice_start = max(0, train_end - batch_start)
                slice_end = min(batch_len, val_end - batch_start)
                val_df = df_batch.slice(slice_start, slice_end - slice_start)
                if len(val_df) > 0:
                    writers['val'].write_table(val_df.to_arrow())

            if batch_end > val_end:
                slice_start = max(0, val_end - batch_start)
                test_df = df_batch.slice(slice_start, batch_len - slice_start)
                if len(test_df) > 0:
                    writers['test'].write_table(test_df.to_arrow())

            current_row = batch_end
        print("\nSplit complete")
    finally:
        for writer in writers.values():
            writer.close()

if __name__ == '__main__':
    input_file = Path("data/normalized/2026-04-26/lob20.parquet")
    output_dir = Path("data/splits/2026-04-26")
    
    split(input_file, output_dir, TRAIN_RATIO, VAL_RATIO)