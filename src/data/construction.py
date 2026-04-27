from sortedcontainers import SortedDict
import json
import polars as pl
from pathlib import Path
from itertools import islice, batched
import numpy as np
import argparse
from typing import Iterator, Iterable
import pyarrow.parquet as pq

LEVELS = 20
OFI_LEVELS = 10
BATCH_SIZE = 300_000

def parse_args():
    parser = argparse.ArgumentParser(description='Script for constructing data from snapshot and stream')
    parser.add_argument('--date', '-d', required=True, help='Date used to find the snapshot and stream records')

    return parser.parse_args()

def compute_ofi_per_level(
        prev_bids: list[tuple[float, float]], 
        prev_asks: list[tuple[float, float]], 
        curr_bids: list[tuple[float, float]], 
        curr_asks: list[tuple[float, float]], 
        ofi_levels: int
    ) -> list[float]:
    """
    Compute OFI per order level from previous and current levels,
    each containing price and volume.

    prev_bids: previous price-volume levels on bid side.
    prev_asks: previous price-volume levels on ask side.
    curr_bids: current price-volume levels on bid side.
    curr_asks: current price-volume levels on ask side.
    ofi_levels: number of OFI levels to compute.
    """
    ofi =[]
    for i in range(ofi_levels):
        # Bid OFI
        if i < len(curr_bids) and i < len(prev_bids):
            cp, cq = curr_bids[i]
            pp, pq = prev_bids[i]
            if cp > pp:    
                bid_delta = cq
            elif cp == pp: 
                bid_delta = cq - pq
            else:          
                bid_delta = -pq
        elif i < len(curr_bids):    # Previous was empty, current exists
            bid_delta = curr_bids[i][1]
        elif i < len(prev_bids):    # Current is empty, previous exists
            bid_delta = -prev_bids[i][1]
        else:
            bid_delta = 0.0     # Both empty
        # Ask OFI
        if i < len(curr_asks) and i < len(prev_asks):
            cp, cq = curr_asks[i]
            pp, pq = prev_asks[i]
            if cp < pp:    
                ask_delta = cq
            elif cp == pp: 
                ask_delta = cq - pq
            else:          
                ask_delta = -pq
        elif i < len(curr_asks):
            ask_delta = curr_asks[i][1]
        elif i < len(prev_asks):
            ask_delta = -prev_asks[i][1]
        else:
            ask_delta = 0.0
            
        ofi.append(bid_delta - ask_delta)
        
    return ofi

def compute_mid_spread(best_ask: float, best_bid: float) -> tuple[float, float]:
    """Compute mid price and spread from best_ask and best_bid."""
    if best_ask <= best_bid:
        raise ValueError(
            f"Crossed spread: best_ask={best_ask}, best_bid={best_bid}"
        )
    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    return mid, spread

def update(
        event: dict,
        bids: SortedDict, 
        asks: SortedDict
    ):
    """
    Update LOB state with new event.

    event: event to update with.
    bids: price-volume map for bids, maintains LOB state.
    asks: price-volume map for asks, maintains LOB state.
    """
    for price, qty in event['b']:
        key = -float(price)
        fqty = float(qty)
        if fqty == 0.0: # If volume is 0, pop the price key
            bids.pop(key, None)
        else:
            bids[key] = fqty
    for price, qty in event['a']:
        key = float(price)
        fqty = float(qty)
        if fqty == 0.0: # If volume is 0, pop the price key
            asks.pop(key, None)
        else:
            asks[key] = fqty

def read_snapshot(snapshot_path: Path) -> dict:
    """
    Read LOB snapshot.

    snapshot_path: path to snapshot.json file.
    """
    with open(snapshot_path) as f:
        snapshot = json.load(f)
    if snapshot['bids'] and snapshot['asks']:
        snapshot_best_ask = float(snapshot['asks'][0][0])
        snapshot_best_bid = float(snapshot['bids'][0][0])
        if snapshot_best_ask <= snapshot_best_bid:
            raise ValueError(
                f"Snapshot has crossed spread: best_ask={snapshot_best_ask}, best_bid={snapshot_best_bid}"
            )
    else:
        raise RuntimeError(
            f"Snapshot {'bids' if not snapshot['bids'] else 'asks'} are empty"
        )
    return snapshot

def iter_events(
        last_update_id: int, 
        bids: SortedDict,
        asks: SortedDict,
        stream_path: Path,
        levels: int, 
    ) -> Iterator[tuple[dict, SortedDict, SortedDict]]:
    """
    Yield (event, top_bids, top_asks) for each stream event.

    last_update_id:
        id of the last snapshot update, this id is used to find
        the correct bridging event from the stream of events.
    bids: price-volume map for bids, maintains LOB state.
    asks: price-volume map for asks, maintains LOB state.
    stream_path: path to stream.jsonl file.
    levels: number of LOB levels to maintain.
    """
    bridged = False
    event: dict | None = None
    # Previous timestamp holder to check correct time flow, see: https://dev.binance.vision/t/event-time-not-sequential-in-diff-depth-stream/1560
    prev_ts: int | None = None
    try:
        with open(stream_path) as f:
            for line in f:
                event = json.loads(line)
                # Discard all old events
                if event['u'] <= last_update_id:
                    continue

                if not bridged:
                    # First event bridging check, snapshot last update must be within event boundaries
                    if not (event['U'] <= last_update_id + 1 <= event['u']):
                        raise RuntimeError(f"First event does not bridge snapshot")
                    bridged = True
                else:
                    # Previous event must end where the next starts
                    if event['U'] != last_update_id + 1:
                        raise RuntimeError(f"Gap detected: expected U={last_update_id + 1}, got U={event['U']}")

                ts = event['E']
                if prev_ts is not None and ts < prev_ts:
                    raise RuntimeError(f"Reversed timestamps: {ts} current < {prev_ts} previous")
                prev_ts = ts
                
                update(event, bids, asks)
                if not bids or not asks:
                    raise RuntimeError(
                        f"Empty {'bids' if not bids else 'asks'} after update"
                    )
                # If update is successful, reset snapshot update id
                last_update_id = event['u']

                top_bids = [(-k, v) for k, v in islice(bids.items(), levels)] 
                top_asks = list(islice(asks.items(), levels))

                yield event, top_bids, top_asks
                
        if not bridged:
            raise RuntimeError('Stream ended without bridging')
    except (IndexError, ValueError) as e:
        ts = event.get('E', 'unknown') if event is not None else 'event is None'
        raise RuntimeError(f"Stream iteration errored at event timestamp: {ts}") from e

def make_row(
        timestamp: int,
        top_bids: list[tuple[float, float]], 
        top_asks: list[tuple[float, float]], 
        prev_bids: list[tuple[float, float]],
        prev_asks: list[tuple[float, float]],
        levels: int, 
        ofi_levels: int
    ) -> list[float]:
    """
    Create a row of the form:
    (event_timestamp, bid_price_1, ..., bid_price_N, ask_price_1, ..., ask_price_N, ofi_1, ..., ofi_M, mid, spread),
    where N and M are equal to levels and ofi_levels, ofi_i is calculated for order level i. 

    timestamp: timestamp of the associated event.
    top_bids: top bid levels after the event sorted by price.
    top_asks: top ask levels after the event sorted by price.
    prev_bids: top bid levels before the event sorted by price.
    prev_asks: top ask levels before the event sorted by price.
    levels: number of order levels.
    ofi_levels: number of OFI levels.
    """
    best_ask = top_asks[0][0]
    best_bid = top_bids[0][0]
    mid, spread = compute_mid_spread(best_ask, best_bid)
    ofi = compute_ofi_per_level(prev_bids, prev_asks, top_bids, top_asks, ofi_levels)

    padded_bids = top_bids + [(np.nan, np.nan)] * (levels - len(top_bids))
    padded_asks = top_asks + [(np.nan, np.nan)] * (levels - len(top_asks))

    # Cluster features for better compression
    row = [timestamp]
    row.extend(float(p) for p, _ in padded_bids)
    row.extend(float(q) for _, q in padded_bids)
    row.extend(float(p) for p, _ in padded_asks)
    row.extend(float(q) for _, q in padded_asks)
    row.extend(ofi)
    row.extend([mid, spread])

    return row

def build_lob(
        snapshot_path: Path, 
        stream_path: Path, 
        levels: int = LEVELS, 
        ofi_levels: int = OFI_LEVELS
    ) -> Iterator[list[float]]:
    """
    Yield rows for LOB in the form:
    (event_timestamp, bid_price_1, ..., bid_price_N, ask_price_1, ..., ask_price_N, ofi_1, ..., ofi_M, mid, spread),
    where N and M are equal to levels and ofi_levels, ofi_i is calculated for order level i.

    snapshot_path: path to snapshot.json file with LOB snapshot.
    stream_path: path to stream.jsonl file with diff streams, used to modify snapshot.
    levels: number of order levels to maintain.
    ofi_levels: number of OFI levels to maintain.
    """
    snapshot = read_snapshot(snapshot_path)
    bids = SortedDict()
    asks = SortedDict()

    for price, qty in snapshot['bids']:
        bids[-float(price)] = float(qty)
    for price, qty in snapshot['asks']:
        asks[float(price)] = float(qty)

    # Keep snapshot bids, asks for future ofi computation
    prev_bids = [(-k, v) for k, v in islice(bids.items(), ofi_levels)]
    prev_asks = list(islice(asks.items(), ofi_levels))

    events = iter_events(snapshot['lastUpdateId'], bids, asks, stream_path, levels)

    for event, top_bids, top_asks in events:
        yield make_row(
            event['E'], # event timestamp
            top_bids, 
            top_asks, 
            prev_bids,
            prev_asks,
            levels, 
            ofi_levels,
        )
        prev_bids = top_bids[:ofi_levels]
        prev_asks = top_asks[:ofi_levels]

def batch_to_arrow(
        batch: list[list[float]], 
        schema: dict, 
    ) -> pq.Table:
    """Convert the batch to Pyarrow Table with the schema."""
    df = pl.DataFrame(batch, schema=schema, orient='row')

    df = df.with_columns(
        pl.when(pl.col(pl.Float64).is_nan())
        .then(None)
        .otherwise(pl.col(pl.Float64))
        .name
        .keep()
    )
    return df.to_arrow()

def save_parquet(
        rows: Iterable[list[float]], 
        output_file: Path, 
        levels: int = LEVELS, 
        ofi_levels: int = OFI_LEVELS, 
        batch_size: int = BATCH_SIZE
    ):
    """
    Write LOB rows of the form:
    (event_timestamp, bid_price_1, ..., bid_price_N, ask_price_1, ..., ask_price_N, ofi_1, ..., ofi_M, mid, spread),
    where N and M are equal to levels and ofi_levels, to parquet files in batches.

    rows: iterable for rows to be written in the file.
    output_file: path to the output parquet file.
    levels: number of order levels rows iterable contains.
    ofi_levels: number of OFI levels rows iterable contains.
    batch_size: number of rows per parquet file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Cluster features for better compression
    cols = ['timestamp']
    cols.extend(f"bid_p{i}" for i in range(levels))
    cols.extend(f"bid_q{i}" for i in range(levels))
    cols.extend(f"ask_p{i}" for i in range(levels))
    cols.extend(f"ask_q{i}" for i in range(levels))
    cols.extend(f"ofi_{i}" for i in range(ofi_levels))
    cols.extend(['mid', 'spread'])

    schema = {'timestamp': pl.Int64}
    schema.update({c: pl.Float64 for c in cols if c != 'timestamp'})

    writer = None
    try:
        print(f"Starting to write {output_file}...")
        for i, batch in enumerate(batched(rows, batch_size), start=1):
            arrow_table = batch_to_arrow(batch, schema)
            if writer is None:
                writer = pq.ParquetWriter(output_file, arrow_table.schema, compression='zstd')
            writer.write_table(arrow_table)
            print(f"Wrote chunk {i} ({len(batch)} rows) -> {output_file.name}")
    finally:          
        if writer is not None:
            writer.close()
            print(f"Finished writing {output_file}.")
        else:
            print('Nothing was written.')
        

if __name__ == '__main__':
    args = parse_args()
    stream_path = Path(f"data/raw/{args.date}/stream.jsonl")
    snapshot_path = Path(f"data/raw/{args.date}/snapshot.json")
    output_file = Path(f"data/books/{args.date}/lob20.parquet")

    rows = build_lob(snapshot_path, stream_path)
    save_parquet(rows, output_file)

