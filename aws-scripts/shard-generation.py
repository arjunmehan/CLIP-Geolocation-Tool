import webdataset as wds
import pandas as pd
import s3fs
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# -------------------
# CONFIG
# -------------------
MANIFEST = "s3://geo-clip-project/metadata/clean_manifest.parquet"
OUTPUT_SHARDS = "s3://geo-clip-project/webdataset/train/train-%06d.tar"
SHARD_SIZE = 5000

fs = s3fs.S3FileSystem(anon=False)

df = pd.read_parquet(MANIFEST, filesystem=fs)

with wds.ShardWriter(OUTPUT_SHARDS, maxcount=SHARD_SIZE) as sink:
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            with fs.open(row["image_s3_path"], "rb") as f:
                img = Image.open(f).convert("RGB")

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=95)
            buf.seek(0)

            key = f"{idx:09d}"

            sink.write({
                "__key__": key,
                "jpg": buf.read(),
                "loc.txt": row["loc_caption"],
                "climate.txt": row["climate_caption"],
                "traffic.txt": row["traffic_caption"],
            })

        except Exception:
            continue
