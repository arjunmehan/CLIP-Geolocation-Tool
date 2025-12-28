import pandas as pd
import boto3
import requests
from PIL import Image
from io import BytesIO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

# -------------------
# CONFIG
# -------------------
INPUT_CSV = "raw_csv/mp16_w_index.csv"
BUCKET = "my-geolocation-clip-project"
IMAGE_PREFIX = "images"
TIMEOUT = 5
MAX_IMAGE_SIZE = (512, 512)
NUM_WORKERS = max(cpu_count() - 1, 1)

# -------------------
# WORKER FUNCTION
# -------------------
def process_row(args):
    idx, row = args
    s3 = boto3.client("s3")  # one per process

    try:
        # Download
        r = requests.get(row["URL"], timeout=TIMEOUT)
        r.raise_for_status()

        # Decode image
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.thumbnail(MAX_IMAGE_SIZE)

        # Encode JPEG
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        shard = idx % 100
        s3_key = f"{IMAGE_PREFIX}/{shard:02d}/{idx:09d}.jpg"

        # Upload
        s3.upload_fileobj(buf, BUCKET, s3_key)

        # Success record
        return {
            "status": "ok",
            "record": {
                "image_s3_path": f"s3://{BUCKET}/{s3_key}",
                "loc_caption": row["loc_caption"],
                "climate_caption": row["cli_caption"],
                "traffic_caption": row["tra_caption"],
            },
        }

    except Exception as e:
        # Failure record
        return {
            "status": "fail",
            "record": {
                "image_url": row["image_url"],
                "error": str(e),
            },
        }

# -------------------
# MAIN
# -------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    inputs = list(df.iterrows())
    successes = []
    failures = []

    print(f"Starting parallel download with {NUM_WORKERS} workers")

    with Pool(NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_row, inputs), total=len(inputs)):
            if result["status"] == "ok":
                successes.append(result["record"])
            else:
                failures.append(result["record"])

    # Write outputs
    clean_df = pd.DataFrame(successes)
    failed_df = pd.DataFrame(failures)

    clean_df.to_parquet("clean_manifest.parquet")
    failed_df.to_csv("failed_urls.csv", index=False)

    # Upload outputs to S3
    s3 = boto3.client("s3")
    s3.upload_file(
        "clean_manifest.parquet",
        BUCKET,
        "metadata/clean_manifest.parquet"
    )
    s3.upload_file(
        "failed_urls.csv",
        BUCKET,
        "metadata/failed_urls.csv"
    )

    print("Stage A complete")
    print(f"Valid images: {len(clean_df)}")
    print(f"Failed URLs: {len(failed_df)}")

if __name__ == "__main__":
    main()
