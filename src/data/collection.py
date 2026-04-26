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
    base_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime('%H%M%S')
    tqdm.write(f"New run_id: {run_id}.")

    snapshot_path = base_dir / f"snapshot_{run_id}.json"
    stream_path = base_dir / f"stream_{run_id}.jsonl"

    uri = "wss://stream.binance.com:9443/ws"

    tqdm.write('Connecting websocket...')
    async with websockets.connect(uri) as websocket:
        # Subscribe
        await websocket.send(json.dumps(subscribe_msg))
        response = await websocket.recv()
        tqdm.write(f"Subscription response: {response}")

        tqdm.write('Initializing queue...')
        msg_queue = asyncio.Queue()
        tqdm.write('Creating websocket reading task...')
        reader_task = asyncio.create_task(ws_reader(websocket, msg_queue))

        first_msg = await msg_queue.get()
        first_event = json.loads(first_msg)
        first_U = first_event['U']

        tqdm.write('Trying to fetch snapshot...')
        async with aiohttp.ClientSession() as session:
            async with asyncio.timeout(30):
                while True:
                    snapshot = await fetch_snapshot(session)
                    snapshot_last_id = snapshot['lastUpdateId']
                    if snapshot_last_id >= first_U:
                        break
                    tqdm.write(f"Snapshot too old ({snapshot_last_id} < {first_U}). Retrying in 2s...")
                    await asyncio.sleep(2) 

        tqdm.write(f"Snapshot acquired, lastUpdateId: {snapshot_last_id}")

        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f)
        tqdm.write('Snapshot saved.')

        tqdm.write('Bridging stream...')
        prev_u = None
        pending = [first_msg]
        async with aiofiles.open(stream_path, 'a') as f:
            count = 0
            pbar = tqdm(total=BATCH_SIZE, desc='Collecting messages', unit='msg')

            while count < BATCH_SIZE:
                if pending:
                    msg = pending.pop(0)
                else:
                    async with asyncio.timeout(3):
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
                    tqdm.write(f"Bridged, first event: U={current_U}, u={current_u}")
                else:
                    if current_U != prev_u + 1:
                        tqdm.write(f"Gap detected: expected U={prev_u + 1}, got U={current_U}. Canceling...")
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
            tqdm.write(f"Websocket closed, ({e}). Retrying in 1s...")
            await asyncio.sleep(1)
        except TimeoutError:
            tqdm.write('Timed out. Retrying in 1s...')
            await asyncio.sleep(1)
        except ConnectionError as e:
            tqdm.write(f"Network connection lost ({type(e).__name__}). Retrying in 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            tqdm.write(f"Unexpected error: {type(e).__name__}: {e}. Retrying in 5s...")
            await asyncio.sleep(5)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        tqdm.write("\nStopped by user")