import os
import boto3
from botocore.exceptions import ClientError


def upload_contract_folders_to_s3(local_root_path: str, bucket_name: str, base_prefix: str = "contracts/legal/"):
    """
    Reads all folders inside `local_root_path` and uploads all PDF & DOCX files
    to S3 under `contracts/legal/<foldername>/...`, preserving subfolder structure.

    local_root_path: path that contains your folders (e.g. "C:/contracts" or "/home/user/contracts")
    bucket_name: your S3 bucket name
    base_prefix: base path in the bucket (default "contracts/legal/")
    """
    s3 = boto3.client("s3")

    # Ensure base_prefix ends with '/'
    if not base_prefix.endswith("/"):
        base_prefix = base_prefix + "/"

    # Iterate over all items in the root directory
    for entry in os.scandir(local_root_path):
        if not entry.is_dir():
            # We only care about folders
            continue

        folder_name = entry.name
        local_folder_path = entry.path

        # S3 "folder" prefix for this local folder
        s3_folder_prefix = f"{base_prefix}{folder_name}/"

        # 1) Create the "folder" in S3 (zero-byte object with trailing slash)
        try:
            print(f"Creating folder in S3: s3://{bucket_name}/{s3_folder_prefix}")
            s3.put_object(Bucket=bucket_name, Key=s3_folder_prefix)
        except ClientError as e:
            print(f"Failed to create folder {s3_folder_prefix} in bucket {bucket_name}: {e}")
            continue  # skip this folder if we can't create it

        # 2) Walk the local folder and upload .pdf and .docx files
        for dirpath, _, filenames in os.walk(local_folder_path):
            # Compute subpath relative to the folder root
            rel_subdir = os.path.relpath(dirpath, local_folder_path)

            # Build the S3 prefix for this subdir
            if rel_subdir == ".":
                current_s3_prefix = s3_folder_prefix
            else:
                current_s3_prefix = s3_folder_prefix + rel_subdir.replace(os.sep, "/") + "/"

            for filename in filenames:
                # Only pdf and docx files (case-insensitive)
                if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".docx")):
                    continue

                local_file_path = os.path.join(dirpath, filename)
                s3_key = current_s3_prefix + filename

                try:
                    print(f"Uploading {local_file_path} -> s3://{bucket_name}/{s3_key}")
                    # boto3's upload_file signature: upload_file(Filename, Bucket, Key)
                    s3.upload_file(local_file_path, bucket_name, s3_key)
                except ClientError as e:
                    print(f"Failed to upload {local_file_path} to {s3_key}: {e}")


if __name__ == "__main__":
    # Example usage – change these to your actual values
    LOCAL_ROOT_PATH = "/path/to/local/folders"    # e.g. "/home/user/contracts_local"
    BUCKET_NAME = "your-s3-bucket-name"

    upload_contract_folders_to_s3(LOCAL_ROOT_PATH, BUCKET_NAME)
