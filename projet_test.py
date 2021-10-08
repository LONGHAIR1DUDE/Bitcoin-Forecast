import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


import scipy as sp

import xgboost as xgb
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from scipy import optimize 

def convertion_prix(tab):
	temp=tab.replace('.','',1)
	temp=temp.replace(',','.',1)

	return float(temp)


def convertion_vol(elt):
   
    temp=elt.replace('K','',1)
    temp=temp.replace(',','.',1) 
    temp=temp.replace('-','0.0',1)   
    return float(temp)*1000
     

def convertion_variation(elt):
    temp=elt.replace(',','.',1)
    temp=temp.replace('%','',1)
    return float(temp)

#c'est moche 
def formating(elt, dtFrame,conversion_type):
    if conversion_type=='prix':
        for i in range(dtFrame[elt].shape[0]):
            dtFrame[elt][i]=convertion_prix(dtFrame[elt][i])

    elif conversion_type=='volume':
        for i in range(dtFrame[elt].shape[0]):
            dtFrame[elt][i]=convertion_vol(dtFrame[elt][i])

    elif conversion_type=='variation':
        for i in range(dtFrame[elt].shape[0]):
            dtFrame[elt][i]=convertion_variation(dtFrame[elt][i])


bitcoin = pd.read_csv('BTC-USD.csv',index_col='Date',parse_dates=True)


formating('Ouv.',bitcoin,'prix')
formating('Dernier',bitcoin,'prix')
formating('Plus Haut',bitcoin,'prix')
formating('Plus Bas',bitcoin,'prix')
formating('Vol.',bitcoin,'volume')
formating('Variation %',bitcoin,'variation')

print(bitcoin.head())
#bitcoin['2020']['Plus Haut'].plot(figsize=(15,6))
#bitcoin['2020']['Variation %'].plot(figsize=(15,6))
#plt.show()




cols_to_use = ['Dernier','Ouv.','Plus Haut','Plus Bas','Vol.']
X = bitcoin[cols_to_use]
# Select target
y = bitcoin['Ouv.']

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)


my_model = RandomForestRegressor(random_state=1)
my_model.fit(X_train, y_train)


predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
plt.plot(predictions)
plt.show()
