import pandas as pd

input_path = "https://animeridw98232.blob.core.windows.net/training-data/customer_churn.csv"
df = pd.read_csv(input_path)
df.head()