import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

role = sagemaker.get_execution_role()
session = sagemaker.Session()

processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=session.boto_region_name,
        version="1.2-1",
        py_version="py3",
    ),
    command=["python3"],
    role=role,
    instance_type="ml.t3.2xlarge",
    instance_count=1,
)

processor.run(
    code="s3://amazon-sagemaker-109003217677-us-east-2-4rbg10ahr7c3op/shared/stage-a/downloader.py",
    inputs=[
        ProcessingInput(
            source="s3://my-geolocation-clip-project/raw_csv/mp16_w_index.csv",
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination="s3://my-geolocation-clip-project/metadata",
        )
    ],
    wait=True,
)
