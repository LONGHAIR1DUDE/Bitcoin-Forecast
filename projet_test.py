import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


import scipy as sp

import xgboost as xgb
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def convertion(tab):
	temp=tab.replace('.','',1)
	temp=temp.replace(',','.',1)

	return float(temp)

def formating(elt):
    for i in range(bitcoin[elt].shape[0]):
    	bitcoin[elt][i]=convertion(bitcoin[elt][i])


bitcoin = pd.read_csv('BTC-USD.csv',index_col='Date',parse_dates=True)


formating('Ouv.')
formating('Dernier')
formating('Plus Haut')
formating('Plus Bas')

print(bitcoin.head())
#bitcoin['2020']['Plus Haut'].plot(figsize=(15,6))

def convertion_vol(elt):
   
    temp=elt.replace('K','',1)
    temp=temp.replace(',','.',1) 
    temp=temp.replace('-','0.0',1)
    
    return float(temp)*1000
for i in range(bitcoin['Vol.'].shape[0]):
    	bitcoin['Vol.'][i]=convertion_vol(bitcoin['Vol.'][i])
     
cols_to_use = ['Dernier','Ouv.','Plus Haut','Plus Bas','Vol.']
X = bitcoin[cols_to_use]
# Select target
y = bitcoin['Ouv.']

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)


my_model = RandomForestRegressor(random_state=1)
my_model.fit(X_train, y_train)


predictions = my_model.predict(X_valid)
plt.plot(predictions)
plt.show()
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
