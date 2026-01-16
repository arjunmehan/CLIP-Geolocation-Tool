import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import subprocess
import sys
from multiprocessing import Pool, cpu_count, Manager
import math
import os

# -------------------
# INSTALL DEPS (fallback-safe)
# -------------------
def install_dependencies():
    packages = ['s3fs', 'webdataset']
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_dependencies()

import s3fs
import webdataset as wds

# -------------------
# CONFIG
# -------------------
MANIFEST = "/opt/ml/processing/input/clean_manifest.parquet"
OUTPUT_PATTERN = "/opt/ml/processing/output/train-%06d.tar"
SHARD_SIZE = 5000
NUM_WORKERS = min(cpu_count(), 8)  # good S3 parallelism

# -------------------
# LOAD MANIFEST
# -------------------
df = pd.read_parquet(MANIFEST)

# -------------------
# GLOBAL SHARD BASE (KEY FIX)
# -------------------
GLOBAL_SHARD_BASE = 100

print(f"Global shard base = {GLOBAL_SHARD_BASE}")

# -------------------
# SPLIT DATAFRAME
# -------------------
def split_dataframe(df, n):
    chunk_size = math.ceil(len(df) / n)
    return [
        df.iloc[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n)
    ]

chunks = split_dataframe(df, NUM_WORKERS)

# -------------------
# COMPUTE SHARD OFFSETS (GLOBAL)
# -------------------
shard_offsets = []
current = GLOBAL_SHARD_BASE

for chunk in chunks:
    shard_offsets.append(current)
    current += math.ceil(len(chunk) / SHARD_SIZE)

# -------------------
# WORKER FUNCTION
# -------------------
def write_shards(args):
    worker_id, chunk, shard_offset, progress_queue = args

    fs = s3fs.S3FileSystem(anon=False)

    with wds.ShardWriter(
        OUTPUT_PATTERN,
        maxcount=SHARD_SIZE,
        start_shard=shard_offset,
    ) as sink:

        for idx, row in chunk.iterrows():
            try:
                with fs.open(row["image_s3_path"], "rb") as f:
                    img = Image.open(f).convert("RGB")

                buf = BytesIO()
                img.save(buf, format="JPEG", quality=95)
                buf.seek(0)

                sink.write({
                    "__key__": f"{idx:09d}",  # GLOBAL SAMPLE KEY
                    "jpg": buf.read(),
                    "loc.txt": row["loc_caption"],
                    "climate.txt": row["climate_caption"],
                    "traffic.txt": row["traffic_caption"],
                })

            except Exception as e:
                print(f"[Worker {worker_id}] Failed idx={idx}: {e}")

            finally:
                progress_queue.put(1)

# -------------------
# MAIN
# -------------------
if __name__ == "__main__":
    manager = Manager()
    progress_queue = manager.Queue()

    total = len(df)

    print(f"Starting shard generation with {NUM_WORKERS} workers")
    print(f"Expected shard range: {GLOBAL_SHARD_BASE} → {current - 1}")

    with tqdm(total=total, desc="Writing shards") as pbar:
        with Pool(NUM_WORKERS) as pool:
            pool.map_async(
                write_shards,
                [
                    (i, chunks[i], shard_offsets[i], progress_queue)
                    for i in range(NUM_WORKERS)
                ],
            )

            completed = 0
            while completed < total:
                progress_queue.get()
                completed += 1
                pbar.update(1)

    print("Shard generation complete")
