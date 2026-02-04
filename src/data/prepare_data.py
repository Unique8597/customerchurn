# prepare_data.py
import os
from storage_utils import upload_if_not_exists, get_blob_uri

container = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
blob_name = os.getenv("AZURE_STORAGE_BLOB_NAME")
local_csv = "data/customerchurn.csv"

# Upload if it doesn't exist
upload_if_not_exists(container, local_csv, blob_name)