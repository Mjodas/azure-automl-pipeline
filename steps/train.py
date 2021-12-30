import argparse
import os

import joblib
import pandas as pd
from azureml.core import Run
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument('--training-data', type=str,
                    help='The training data', dest='training_data')

parser.add_argument('--learning_rate', type=float,
                    dest='learning_rate', default=0.1, help='learning rate')

parser.add_argument('--n_estimators', type=int, dest='n_estimators',
                    default=100, help='number of estimators')

args = parser.parse_args()

training_data = args.training_data
learning_rate = args.learning_rate
n_estimators = args.n_estimators

run = Run.get_context()

path = os.path.join(training_data, 'data.csv')
df = pd.read_csv(path)

X, y = df.drop('SalePrice', axis='columns').values, df['SalePrice'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=0)

gbr = GradientBoostingRegressor(
    learning_rate=learning_rate, n_estimators=n_estimators)
gbr.fit(X_train, y_train)

predictions = gbr.predict(X_test)

rmse = mean_squared_error(y_test, predictions, squared=True)
r2 = r2_score(y_test, predictions)

run.log('RMSE', rmse)
run.log('R2', r2)

# Save the model in the run outputs
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=gbr, filename='outputs/house_price_model.pkl')

run.complete()
