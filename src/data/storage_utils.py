"""
storage_utils.py

Handles uploading and checking CSV files in Azure Storage using a Service Principal.
Compatible with Azure ML workspace storage.
"""

import os
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv(".env")



def get_blob_service_client():
    """
    Creates BlobServiceClient using a Service Principal.
    Requires the following environment variables:
        AZURE_TENANT_ID
        AZURE_CLIENT_ID
        AZURE_CLIENT_SECRET
        AZURE_STORAGE_ACCOUNT_NAME
    """
    tenant_id = os.getenv("TENANT_ID")
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")

    if not all([tenant_id, client_id, client_secret, account_name]):
        raise ValueError(
            "TENANT_ID, CLIENT_ID, CLIENT_SECRET, AZURE_STORAGE_ACCOUNT_NAME must be set"
        )
    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret
    )

    blob_service_client = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=credential
    )

    return blob_service_client


def upload_if_not_exists(container_name: str, local_file_path: str, blob_name: str):
    """
    Uploads a file to Azure Blob Storage only if it does not exist.

    Args:
        container_name: Name of the container in storage account
        local_file_path: Path to local CSV file
        blob_name: Name to save in blob storage
    Returns:
        True if uploaded, False if already exists
    """
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)

    # Create container if it does not exist
    try:
        container_client.create_container()
        print(f"Created container: {container_name}")
    except Exception:
        pass  # Already exists

    # Check if blob exists
    blob_list = [b.name for b in container_client.list_blobs(name_starts_with=blob_name)]
    if blob_name in blob_list:
        print(f"Blob '{blob_name}' already exists. Using existing file.")
        return False

    # Upload blob
    with open(local_file_path, "rb") as f:
        container_client.upload_blob(name=blob_name, data=f)
        print(f"Uploaded '{local_file_path}' as blob '{blob_name}'")
    return True


def get_blob_uri(container_name: str, blob_name: str):
    """
    Returns the full URI of a blob for Azure ML to consume.
    Requires AZURE_STORAGE_ACCOUNT_NAME environment variable.
    """
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    if not account_name:
        raise ValueError("AZURE_STORAGE_ACCOUNT_NAME environment variable is not set.")

    return f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"