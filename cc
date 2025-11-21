import os
import boto3
from botocore.exceptions import ClientError


def upload_finance_folders_to_s3(
    local_root_path: str,
    bucket_name: str,
    base_prefix: str = "user/tt/finance/",
):
    """
    Upload finance client folders to S3, preserving client + subfolder structure.

    Local layout (example):
      ce/finance/
        client1/
          subA/file1.pdf
          subB/file2.docx
        client2/
          ...

    S3 layout (result):
      s3://<bucket>/user/tt/finance/client1/subA/file1.pdf
      s3://<bucket>/user/tt/finance/client1/subB/file2.docx
      s3://<bucket>/user/tt/finance/client2/...

    Parameters
    ----------
    local_root_path : str
        Path that contains your finance client folders, e.g. "ce/finance".
    bucket_name : str
        Your S3 bucket name.
    base_prefix : str
        Base path in the bucket. Default "user/tt/finance/" matches your setup.
    """
    s3 = boto3.client("s3")

    # Ensure base_prefix ends with '/'
    if not base_prefix.endswith("/"):
        base_prefix = base_prefix + "/"

    # Iterate over all items in the finance root directory (client folders)
    for entry in os.scandir(local_root_path):
        if not entry.is_dir():
            # Only care about client folders
            continue

        client_name = entry.name               # e.g. "client1"
        local_client_path = entry.path         # e.g. "ce/finance/client1"

        # S3 prefix for this client folder, e.g. "user/tt/finance/client1/"
        s3_client_prefix = f"{base_prefix}{client_name}/"

        # 1) (Optional) Create an empty "folder" marker in S3
        try:
            print(f"Creating client folder in S3: s3://{bucket_name}/{s3_client_prefix}")
            s3.put_object(Bucket=bucket_name, Key=s3_client_prefix)
        except ClientError as e:
            print(f"Failed to create folder {s3_client_prefix} in bucket {bucket_name}: {e}")
            # You can continue; folder markers are not strictly required
            # but we'll just skip this client on hard error
            continue

        # 2) Walk the local client folder and upload .pdf and .docx files
        for dirpath, _, filenames in os.walk(local_client_path):
            # dirpath might be:
            #   ce/finance/client1
            #   ce/finance/client1/subA
            #   ce/finance/client1/subB
            # We want the path *inside* client1
            rel_subdir = os.path.relpath(dirpath, local_client_path)
            # rel_subdir == "." for the client root

            if rel_subdir == ".":
                current_s3_prefix = s3_client_prefix  # user/tt/finance/client1/
            else:
                # user/tt/finance/client1/subA/
                current_s3_prefix = s3_client_prefix + rel_subdir.replace(os.sep, "/") + "/"

            for filename in filenames:
                # Only pdf and docx files (case-insensitive)
                lower_name = filename.lower()
                if not (lower_name.endswith(".pdf") or lower_name.endswith(".docx")):
                    continue

                local_file_path = os.path.join(dirpath, filename)
                s3_key = current_s3_prefix + filename

                try:
                    print(f"Uploading {local_file_path} -> s3://{bucket_name}/{s3_key}")
                    s3.upload_file(local_file_path, bucket_name, s3_key)
                except ClientError as e:
                    print(f"Failed to upload {local_file_path} to s3://{bucket_name}/{s3_key}: {e}")


if __name__ == "__main__":
    # Change these to your actual values
    # Point this at your local finance root: ce/finance
    LOCAL_FINANCE_ROOT = "ce/finance"          # or full path like "/opt/ml/input/ce/finance"
    BUCKET_NAME = "your-s3-bucket-name"

    # base_prefix "user/tt/finance/" -> s3://<bucket>/user/tt/finance/<client>/...
    upload_finance_folders_to_s3(
        local_root_path=LOCAL_FINANCE_ROOT,
        bucket_name=BUCKET_NAME,
        base_prefix="user/tt/finance/",
    )
