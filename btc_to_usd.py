import pandas as pd
import numpy as np
import scipy as sp
import pandas as pd
import xgboost as xgb
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

data = pd.read_csv('BTC-USD.csv')
cols_to_use = ['Date','Dernier','Ouv.','Plus Haut','Plus Bas','Vol.','Variation %']
X = data[cols_to_use]
# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)


my_model = XGBRegressor()
my_model.fit(X_train, y_train)


predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

    