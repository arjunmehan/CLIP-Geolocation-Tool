import pandas as pd
import boto3
import requests
from PIL import Image
from io import BytesIO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
from botocore.exceptions import ClientError

# -------------------
# CONFIG
# -------------------
INPUT_CSV = "/opt/ml/processing/input/mp16_w_index.csv"
OUTPUT_DIR = "/opt/ml/processing/output"

BUCKET = "my-geolocation-clip-project"
IMAGE_PREFIX = "images"
TIMEOUT = 5
MAX_IMAGE_SIZE = (512, 512)
NUM_WORKERS = max(cpu_count() - 1, 1)

# -------------------
# HELPERS
# -------------------
def s3_object_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

# -------------------
# WORKER FUNCTION
# -------------------
def process_row(args):
    idx, row = args
    s3 = boto3.client("s3")

    shard = idx % 100
    s3_key = f"{IMAGE_PREFIX}/{shard:02d}/{idx:09d}.jpg"

    try:
        # -------------------
        # RESUME-SAFE CHECK
        # -------------------
        if s3_object_exists(s3, BUCKET, s3_key):
            return {
                "status": "ok",
                "record": {
                    "image_s3_path": f"s3://{BUCKET}/{s3_key}",
                    "loc_caption": row["loc_caption"],
                    "climate_caption": row["cli_caption"],
                    "traffic_caption": row["tra_caption"],
                },
            }

        # -------------------
        # DOWNLOAD IMAGE
        # -------------------
        r = requests.get(row["URL"], timeout=TIMEOUT)
        r.raise_for_status()

        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.thumbnail(MAX_IMAGE_SIZE)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        # -------------------
        # UPLOAD TO S3
        # -------------------
        s3.upload_fileobj(buf, BUCKET, s3_key)

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
        return {
            "status": "fail",
            "record": {
                "image_url": row["URL"],
                "error": str(e),
            },
        }

# -------------------
# MAIN
# -------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # -------------------
    # WRITE OUTPUTS
    # -------------------
    clean_df = pd.DataFrame(successes)
    failed_df = pd.DataFrame(failures)

    clean_df.to_parquet(f"{OUTPUT_DIR}/clean_manifest.parquet")
    failed_df.to_csv(f"{OUTPUT_DIR}/failed_urls.csv", index=False)

    print("Stage A complete")
    print(f"Valid images: {len(clean_df)}")
    print(f"Failed URLs: {len(failed_df)}")

if __name__ == "__main__":
    main()
