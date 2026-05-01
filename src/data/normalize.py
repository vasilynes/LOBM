import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Iterable
import pyarrow.parquet as pq
import argparse 
from . import helpers

WINDOW = 10_000
BATCH_SIZE = 300_000
LEVEL = 20
OFI_LEVELS = 10

PRICE_COLS = [f"bid_p{i}" for i in range(LEVEL)] + [f"ask_p{i}" for i in range(LEVEL)]
QTY_COLS = [f"bid_q{i}" for i in range(LEVEL)] + [f"ask_q{i}" for i in range(LEVEL)]
OFI_COLS = [f"ofi_{i}" for i in range(OFI_LEVELS)]
STAT_COLS = ['spread']

NORM_COLS = PRICE_COLS + QTY_COLS + OFI_COLS + STAT_COLS

def parse_args():
    parser = argparse.ArgumentParser(description='Script for normalizing limit order data')
    parser.add_argument('--date', '-d', required=True, help='Date of the LOB folder')
    parser.add_argument('--time', '-t', required=True, help='Time of the LOB data')

    return parser.parse_args()

def rolling_zscore(lf: pl.LazyFrame, cols: list, window_size: int) -> pl.LazyFrame:
    """
    Build a computational graph for sliding a rolling window 
    along given columns and normalizing the data frame.
    """
    zscore_exprs = []
    for col in cols:
        mean = (
            pl.col(col)
            .rolling_mean(
                window_size=window_size,
                min_samples=2
            )
            .shift(1)
        )
        std = (
            pl.col(col)
            .rolling_std(
                window_size=window_size,
                min_samples=2
            )
            .shift(1)
        )

        zscore_exprs.append(
            pl.when(std.is_null())
            .then(None)
            .when(std < 1e-8) # if window is "constant"
            .then(pl.lit(0.0))  # replace with column of zeros
            .otherwise((pl.col(col) - mean) / std)
            .cast(pl.Float32)
            .alias(col)
        )

    return lf.with_columns(zscore_exprs)

def normalize(
        batches: Iterable[pl.DataFrame], 
        output_file: Path, 
        window_size: int,
        norm_cols: list
    ):
    """
    Normalize columns norm_cols using rolling 
    window seamlessly sliding along batches.
    Write data to output_file in batches.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tail = None
    writer = None
    try:
        print('Starting normalization...')
        for i, raw_batch in enumerate(batches):
            raw_lf = raw_batch.lazy()

            if tail is not None:
                lf = pl.concat([tail.lazy(), raw_lf])
            else:
                lf = raw_lf

            normalized = rolling_zscore(lf, norm_cols, window_size)

            if tail is not None:
                normalized = normalized.slice(len(tail))    # slice away the tail
            else:
                normalized = normalized.slice(window_size)  # slice away the warmup rows

            chunk = normalized.collect()

            null_count = chunk.select(
                pl.sum_horizontal(pl.col(norm_cols).null_count())
                ).item()

            if null_count > 0:
                print(f"Error, non-warmup nulls detected: {null_count} total")
                print('Batch discarded, stopping normalization...')
                raise ValueError('Unexpected nulls in normalized data')
            
            arrow_table = chunk.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    output_file, 
                    arrow_table.schema, 
                    compression='zstd'
                )
            writer.write_table(arrow_table)
            print(f"Wrote chunk {i} ({len(chunk)} rows) -> {output_file.name}")

            tail = raw_batch.tail(window_size)
    finally:
        if writer is not None:
            writer.close()
            print(f"Finished normalization, wrote to {output_file}.")
        else:
            print('Nothing was written.')

if __name__ == '__main__':
    today = datetime.now().strftime('%Y-%m-%d')
    args = parse_args()
    input_file = Path(f"data/books/{args.date}/lob20_{args.time}.parquet")
    output_file = Path(f"data/normalized/{args.date}/lob20_{args.time}.parquet")
    batches = helpers.iter_batches(input_file, BATCH_SIZE)
    normalize(batches, output_file, WINDOW, NORM_COLS)
