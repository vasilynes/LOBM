from sortedcontainers import SortedDict
import json
import polars as pl
from pathlib import Path
from datetime import datetime
from itertools import islice
import numpy as np

LEVEL = 20
OFI_LEVELS = 10
BATCH_SIZE = 300000
STRICT = True

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

def build_lob(snapshot_path, stream_path):
    with open(snapshot_path) as f:
        snapshot = json.load(f)

    bids = SortedDict()
    asks = SortedDict()

    for price, qty in snapshot['bids']:
        bids[-float(price)] = float(qty)
    for price, qty in snapshot['asks']:
        asks[float(price)] = float(qty)

    last_update_id = snapshot['lastUpdateId']

    prev_bids = [(-k, v) for k, v in islice(bids.items(), OFI_LEVELS)]
    prev_asks = list(islice(asks.items(), OFI_LEVELS))

    bridged = False
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

            update(bids, asks, event)
            last_update_id = event['u']

            top_bids = [(-k, v) for k, v in islice(bids.items(), LEVEL)] 
            top_asks = list(islice(asks.items(), LEVEL))

            try:
                mid = (top_bids[0][0] + top_asks[0][0]) / 2
            except (IndexError, TypeError) as e:
                print(f"Mid price calculation errored, event timestamp: {event.get('E', 'unknown')}, error: {e}")
                if STRICT:
                    raise 
                else:
                    print('Defaulting to null...')
                    mid = None

            try:
                spread = top_asks[0][0] - top_bids[0][0]
            except (IndexError, TypeError) as e:
                print(f"Spread calculation errored, event timestamp: {event.get('E', 'unknown')}, error: {e}")
                if STRICT:
                    raise 
                else:
                    print('Defaulting to null...')
                    spread = None

            try:
                ofi = compute_ofi_per_level(prev_bids, prev_asks, top_bids, top_asks)
            except (IndexError, TypeError) as e:
                print(f"OFI calculation errored, event timestamp: {event.get('E', 'unknown')}, error: {e}")
                if STRICT:
                    raise 
                print('Defaulting to null...')
                ofi = [None] * OFI_LEVELS
            finally:
                prev_bids = top_bids[:OFI_LEVELS]
                prev_asks = top_asks[:OFI_LEVELS]

            while len(top_bids) < LEVEL:
                top_bids.append((np.nan, np.nan))
            while len(top_asks) < LEVEL:
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

def save_parquet(records_generator, output_dir, level=LEVEL):
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
    cols += [f"ofi_{i}" for i in range(OFI_LEVELS)]
    cols.extend(['mid', 'spread'])

    for row in records_generator:
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            write_parquet(batch, cols, file_idx, output_dir)
            batch = []
            file_idx += 1

    if batch:
        write_parquet(batch, cols, file_idx, output_dir)

if __name__ == '__main__':
    today = datetime.now().strftime('%Y-%m-%d')   
    stream_path = Path(f"data/raw/{today}/stream.jsonl")
    snapshot_path = Path(f"data/raw/{today}/snapshot.json")
    output_dir = Path(f"data/books/{today}")

    rows = build_lob(snapshot_path, stream_path)
    save_parquet(rows, output_dir)

