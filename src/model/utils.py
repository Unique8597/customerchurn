from io import BytesIO
import numpy as np
import joblib
import os
from data.storage_utils import get_blob_service_client

container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

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
