from io import BytesIO
import numpy as np
import joblib
import os
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient


container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

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


blob_service_client = get_blob_service_client()


def load_blob_numpy(blob_path: str):
    """Loads a .npy file from Azure Blob directly into memory."""
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_path
    )

    stream = BytesIO()
    blob_client.download_blob().readinto(stream)
    stream.seek(0)
    return np.load(stream, allow_pickle=True)


def load_blob_joblib(blob_path: str):
    """Loads a joblib file from Azure Blob into memory."""
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_path
    )

    stream = BytesIO()
    blob_client.download_blob().readinto(stream)
    stream.seek(0)
    return joblib.load(stream)
