# download a specific folder from an s3 bucket using boto3
import argparse
import logging
from pathlib import Path

import boto3

from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def download_s3_folder(bucket_name: str, folder_name: str, local_dir: str = "./data") -> None:
    """Downloads a specific folder from an S3 bucket to a local directory using boto3.

    Args:
        bucket_name (str): The name of the S3 bucket to download.
        folder_name (str): The folder inside the S3 bucket to download.
        local_dir (str): The local directory where the bucket contents will be saved.
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    prefix = folder_name.strip("/")
    if prefix:
        prefix = f"{prefix}/"

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    try:
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" not in page:
                logger.warning(
                    f"No objects found in folder '{folder_name}' in bucket '{bucket_name}'"
                )
                return

            for obj in page["Contents"]:
                key = obj["Key"]
                target = local_path / key

                if key.endswith("/") or key == prefix:
                    logger.debug(f"Skipping directory placeholder: {key}")
                    continue
                # Create directories if needed
                target.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"Downloading {key} -> {target}")
                s3.download_file(bucket_name, key, str(target))

        logger.info(
            f"Successfully downloaded folder '{folder_name}' from bucket "
            f"'{bucket_name}' to '{local_dir}'"
        )

    except NoCredentialsError:
        logger.error("AWS credentials not found.")
        raise
    except ClientError as e:
        logger.error(f"AWS client error: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a folder from an S3 bucket using boto3")
    parser.add_argument("--bucket_name", default="osapiens-terra-challenge", help="Name of the S3 bucket")
    parser.add_argument("--folder_name", default="makeathon-challenge", help="Name of the folder inside the S3 bucket")
    parser.add_argument("--local_dir", default="./data", help="Local directory to save files")

    args = parser.parse_args()

    download_s3_folder(args.bucket_name, args.folder_name, args.local_dir)
