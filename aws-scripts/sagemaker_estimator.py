# The dependencies parameter may not be working as expected
# Let's modify train.py to install packages at runtime

import boto3

s3_client = boto3.client('s3')

# Download original train.py from S3
bucket = 'amazon-sagemaker-109003217677-us-east-2-4rbg10ahr7c3op'
key = 'shared/stage-c/requirements.txt'

print("Downloading requirements.txt from S3...")
s3_client.download_file(bucket, key, 'requirements.txt')
print("✓ Downloaded requirements.txt")

key = 'shared/stage-c/train-2.py'
print("Downloading train.py from S3...")
s3_client.download_file(bucket, key, 'train-2.py')
print("✓ Downloaded train-2.py")

# Now start the training job
from sagemaker.pytorch import PyTorch
import sagemaker

session = sagemaker.Session()

estimator = PyTorch(
    entry_point="train-2.py",
    framework_version="1.13",
    py_version="py39",
    instance_count=1,
    instance_type="ml.p3.2xlarge",
    hyperparameters={
        "shards": "/opt/ml/input/data/train/train-{000000..000199}.tar",
        "batch_size": 64,
        "workers": 4,
        "steps": 10000,
        "lr": 5e-5,
        "output_dir": "s3://my-geolocation-clip-project/model/",
    },
)

print("\nStarting training job with runtime package installation...")
estimator.fit({
    "train": "s3://my-geolocation-clip-project/webdataset/train/"
})
