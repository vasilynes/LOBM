from sortedcontainers import SortedDict
import json
import polars as pl
from pathlib import Path
from itertools import islice
import numpy as np
import argparse

LEVELS = 20
OFI_LEVELS = 10
BATCH_SIZE = 300000

def parse_args():
    parser = argparse.ArgumentParser(description='Script for constructing data from snapshot and stream')
    parser.add_argument('--date', '-d', required=True, help='Date used to find the snapshot and stream records')

    return parser.parse_args()

def to_array(levels_list, n):
    arr = np.full((n, 2), np.nan)
    k = min(len(levels_list), n)
    if k > 0:
        arr[:k] = levels_list[:k]
    return arr

def compute_delta(curr, prev, curr_present, prev_present, comparator):
    presence = curr_present & prev_present
    unchanged = curr[:,0] == prev[:,0]
    improved = comparator(curr[:,0], prev[:,0])

    return np.where(
        presence,    # if both levels are non-NaN
        np.where(
            unchanged,  # if price didn't change
            curr[:,1] - prev[:,1],  # take volume delta
            np.where(   
                improved,  # if price improved
                curr[:,1], # take current volume
                -prev[:,1] # else take previous negative volume
            )
        ),
        np.where(   # if one level is NaN
            curr_present,  # if current level is present
            curr[:,1], # take its volume
            np.where(  
                prev_present, # else if previous level is present
                -prev[:,1], # take its negative volume
                np.nan  # otherwise no volume for current level
            )
        )
    )

def compute_ofi_per_level(prev_bids, prev_asks, curr_bids, curr_asks, levels=OFI_LEVELS):
    pb = to_array(prev_bids, levels)    # Pad to fill missing levels
    cb = to_array(curr_bids, levels)
    pa = to_array(prev_asks, levels)
    ca = to_array(curr_asks, levels)

    curr_b_present = ~np.isnan(cb).any(axis=1)  # mark non-NaN levels
    prev_b_present = ~np.isnan(pb).any(axis=1)
    curr_a_present = ~np.isnan(ca).any(axis=1)
    prev_a_present = ~np.isnan(pa).any(axis=1)

    bid_delta = compute_delta(cb, pb, curr_b_present, prev_b_present, lambda x, y: x > y)
    ask_delta = compute_delta(ca, pa, curr_a_present, prev_a_present, lambda x, y: x < y)

    return (bid_delta - ask_delta).tolist()

def compute_mid_spread(best_ask, best_bid):
    if best_ask <= best_bid:
        raise ValueError(
            f"Crossed spread: best_ask={best_ask}, best_bid={best_bid}"
        )
    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    return mid, spread

def update(bids, asks, event):
    for price, qty in event['b']:
        key = -float(price)
        if qty == '0.00000000': # fragile, but standard for BTCUSDT spot
            bids.pop(key, None)
        else:
            bids[key] = float(qty)
    for price, qty in event['a']:
        key = float(price)
        if qty == '0.00000000':
            asks.pop(key, None)
        else:
            asks[key] = float(qty)

def build_lob(snapshot_path, stream_path, levels=LEVELS, ofi_levels=OFI_LEVELS):
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

    bids = SortedDict()
    asks = SortedDict()

    for price, qty in snapshot['bids']:
        bids[-float(price)] = float(qty)
    for price, qty in snapshot['asks']:
        asks[float(price)] = float(qty)

    last_update_id = snapshot['lastUpdateId']

    prev_bids = [(-k, v) for k, v in islice(bids.items(), ofi_levels)]
    prev_asks = list(islice(asks.items(), ofi_levels))

    bridged = False
    event = None
    prev_ts = None
    try:
        with open(stream_path) as f:
            for line in f:
                event = json.loads(line)

                if event['u'] <= last_update_id:
                    continue

                if not bridged:
                    # first event bridging check
                    if not (event['U'] <= last_update_id + 1 <= event['u']):
                        raise RuntimeError(f"First event does not bridge snapshot")
                    bridged = True
                else:
                    if event['U'] != last_update_id + 1:
                        raise RuntimeError(f"Gap detected: expected U={last_update_id + 1}, got U={event['U']}")

                ts = event['E']
                if prev_ts is not None and ts < prev_ts:
                    raise RuntimeError(f"Reversed timestamps: {ts} current < {prev_ts} previous")
                prev_ts = ts

                update(bids, asks, event)
                if not bids or not asks:
                    raise RuntimeError(
                        f"Empty {'bids' if not bids else 'asks'} after update"
                    )
                last_update_id = event['u']

                top_bids = [(-k, v) for k, v in islice(bids.items(), levels)] 
                top_asks = list(islice(asks.items(), levels))

                best_ask = top_asks[0][0]
                best_bid = top_bids[0][0]
                mid, spread = compute_mid_spread(best_ask, best_bid)
                ofi = compute_ofi_per_level(prev_bids, prev_asks, top_bids, top_asks)

                prev_bids = top_bids[:ofi_levels]
                prev_asks = top_asks[:ofi_levels]

                while len(top_bids) < levels:
                    top_bids.append((np.nan, np.nan))
                while len(top_asks) < levels:
                    top_asks.append((np.nan, np.nan))

                row = [event['E']]
                # Cluster features for better compression
                row += [float(p) for p, _ in top_bids]
                row += [float(q) for _, q in top_bids]
                row += [float(p) for p, _ in top_asks]
                row += [float(q) for _, q in top_asks]
                row.extend(ofi)
                row.extend([mid, spread])

                yield row
        if not bridged:
            raise RuntimeError('Stream ended without bridging')
    except (IndexError, ValueError, RuntimeError) as e:
        ts = event.get('E', 'unknown') if event is not None else 'event is None'
        raise RuntimeError(f"Stream iteration errored at event timestamp: {ts}") from e


def write_parquet(batch, cols, file_idx, output_dir):
    schema = {'timestamp': pl.Int64}
    schema.update({c: pl.Float64 for c in cols if c != 'timestamp'})
    df = pl.DataFrame(batch, schema=schema, orient='row')
    df = df.with_columns(
        pl.when(pl.col(pl.Float64).is_nan())
        .then(None)
        .otherwise(pl.col(pl.Float64))
        .name.keep()
    )
    output_path = output_dir / f"lob20_{file_idx:05d}.parquet"
    df.write_parquet(output_path, compression='zstd')
    print(f"Written batch {file_idx}, {len(batch)} rows -> {output_path}")

def save_parquet(records_generator, output_dir, level=LEVELS, ofi_levels=OFI_LEVELS, batch_size=BATCH_SIZE):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = []
    file_idx = 0

    cols = ['timestamp']
    # Cluster features for better compression
    cols += [f"bid_p{i}" for i in range(level)]
    cols += [f"bid_q{i}" for i in range(level)]
    cols += [f"ask_p{i}" for i in range(level)]
    cols += [f"ask_q{i}" for i in range(level)]
    cols += [f"ofi_{i}" for i in range(ofi_levels)]
    cols.extend(['mid', 'spread'])

    for row in records_generator:
        batch.append(row)
        if len(batch) >= batch_size:
            write_parquet(batch, cols, file_idx, output_dir)
            batch = []
            file_idx += 1

    if batch:
        write_parquet(batch, cols, file_idx, output_dir)

if __name__ == '__main__':
    args = parse_args()
    stream_path = Path(f"data/raw/{args.date}/stream.jsonl")
    snapshot_path = Path(f"data/raw/{args.date}/snapshot.json")
    output_dir = Path(f"data/books/{args.date}")

    rows = build_lob(snapshot_path, stream_path)
    save_parquet(rows, output_dir)

