from azureml.core import Dataset, Workspace

data_path = '../data/house_prices.csv'

ws = Workspace.from_config()
default_ds = ws.get_default_datastore()

try:
    # Try to upload the dataset
    default_ds.upload_files([data_path], overwrite=True,
                            target_path='house-prices', show_progress=True)

except Exception as e:
    print(e)


# Create a dataset
tabular_dataset = Dataset.Tabular.from_delimited_files(
    path=(default_ds, 'house-prices/*.csv'))

try:
    # Try to register
    tabular_dataset.register(ws, 'tabular-house-prices', description="House prices",
                             create_new_version=True, tags={'format': 'csv'})
except Exception as e:
    print(e)
