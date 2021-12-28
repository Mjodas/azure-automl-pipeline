import argparse
import os

from azureml.core import Run
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()

parser.add_argument("--input-data", type=str,
                    help="Source data", dest='input_data')
parser.add_argument("--prepared-data", type=str,
                    help="Directory for results", dest='prepared_data')

args = parser.parse_args()

save_folder = args.prepared_data

run = Run.get_context()

house_prices = run.input_datasets['raw_data'].to_pandas_dataframe()
house_prices = house_prices.drop('Id')

labels = house_prices['SalePrice']

# Being lazy, skipping all string columns. 
features = house_prices.select_dtypes(exclude=['str'])

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

house_prices = features
house_prices['SalePrice'] = labels

n_rows = len(house_prices)
run.log('Processed rows', n_rows)

os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, 'data.csv')
house_prices.to_csv(save_path, index=False, header=True)
