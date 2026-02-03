# prepare_data.py
from storage_utils import upload_if_not_exists, get_blob_uri

container = "training-data"
local_csv = "data/customerchurn.csv"
blob_name = "customer_churn.csv"

# Upload if it doesn't exist
upload_if_not_exists(container, local_csv, blob_name)

# Get the blob URI
data_uri = get_blob_uri(container, blob_name)
print(f"Data available at: {data_uri}")