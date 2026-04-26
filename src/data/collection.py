import asyncio
import websockets
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import aiohttp
import aiofiles
from websockets.asyncio.client import ClientConnection

BATCH_SIZE = 300_000
subscribe_msg = {
    'method': 'SUBSCRIBE',
    'params': ['btcusdt@depth@100ms'],
    'id': 1
}

async def ws_reader(websocket: ClientConnection, queue: asyncio.Queue):
    """Read messages from websocket and put them in queue."""
    async for raw in websocket:
        await queue.put(raw)

async def fetch_snapshot(session):
    async with session.get(
        'https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=5000'
    ) as resp:
        return await resp.json()

async def collect_order_book():
    today = datetime.now().strftime('%Y-%m-%d')
    base_dir = Path(f"data/raw/{today}")
    base_dir.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime('%H%M%S')
    snapshot_path = base_dir / f"snapshot_{run_id}.json"
    stream_path = base_dir / f"stream_{run_id}.jsonl"

    uri = "wss://stream.binance.com:9443/ws"

    async with websockets.connect(uri) as websocket:
        # Subscribe
        await websocket.send(json.dumps(subscribe_msg))
        response = await websocket.recv()
        tqdm.write(f"Subscription response: {response}")

        msg_queue = asyncio.Queue()
        reader_task = asyncio.create_task(ws_reader(websocket, msg_queue))

        first_msg = await msg_queue.get()
        first_event = json.loads(first_msg)
        first_U = first_event['U']

        async with aiohttp.ClientSession() as session:
            while True:
                snapshot = await fetch_snapshot(session)
                snapshot_last_id = snapshot['lastUpdateId']
                if snapshot_last_id >= first_U:
                    break
                tqdm.write(f"Snapshot too old ({snapshot_last_id} < {first_U}), retrying...")
                await asyncio.sleep(2) 

        tqdm.write(f"Snapshot lastUpdateId: {snapshot_last_id}")

        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f)

        async with aiofiles.open(stream_path, 'a') as f:
            count = 0
            pbar = tqdm(total=BATCH_SIZE, desc='Collecting messages', unit='msg')

            prev_u = None
            pending = [first_msg]

            while count < BATCH_SIZE:
                if pending:
                    msg = pending.pop(0)
                else:
                    msg = await msg_queue.get()

                event = json.loads(msg)

                if event.get('u') <= snapshot_last_id:
                    continue

                current_U = event['U']
                current_u = event['u']

                if prev_u is None:
                    if not (current_U <= snapshot_last_id + 1 <= current_u):
                        tqdm.write(f"Skipping non-bridging event: U={current_U}, u={current_u}, snapshot={snapshot_last_id}")
                        continue
                    tqdm.write(f"Synchronized, first event: U={current_U}, u={current_u}")
                else:
                    if current_U != prev_u + 1:
                        tqdm.write(f"Gap detected: expected U={prev_u + 1}, got U={current_U}. Re-synchronizing...")
                        reader_task.cancel()
                        return 

                await f.write(msg + '\n')
                prev_u = current_u
                count += 1
                pbar.update(1)
            else:
                tqdm.write('Batch exhausted, finishing...')

            reader_task.cancel()
            pbar.close()

async def main():
    """Run the collector, resyncing on drops."""
    while True:
        try:
            tqdm.write(f"\nStarting collection at {datetime.now()}")
            await collect_order_book()
        except websockets.exceptions.ConnectionClosed as e:
            tqdm.write(f"Websocket closed, ({e}). Resyncing...")
            await asyncio.sleep(1)
        except (ConnectionError, asyncio.TimeoutError) as e:
            tqdm.write(f"Network connection lost ({e.__class__.__name__}). Retrying...")
            await asyncio.sleep(3)
        except Exception as e:
            tqdm.write(f"Unexpected error: {type(e).__name__}: {e}. Retrying...")
            await asyncio.sleep(5)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        tqdm.write("\nStopped by user")