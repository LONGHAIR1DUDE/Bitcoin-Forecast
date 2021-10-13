import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


import scipy as sp

import xgboost as xgb
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn import preprocessing
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
cols_to_use = ['Dernier','Plus Haut','Plus Bas','Vol.','Variation %']



formating('Ouv.',bitcoin,'prix')
formating('Dernier',bitcoin,'prix')
formating('Plus Haut',bitcoin,'prix')
formating('Plus Bas',bitcoin,'prix')
formating('Vol.',bitcoin,'volume')
formating('Variation %',bitcoin,'variation')



for i in cols_to_use:
    bitcoin[i] = bitcoin[i].astype(float)

print(bitcoin)
print(bitcoin.dtypes)
X = bitcoin[cols_to_use]
# Select target
y = bitcoin['Ouv.']


x_train = X['2012':'2020']
x_valid = X['2021']

y_train=y['2012':'2020']
y_valid=y['2021']



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(x_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(x_valid, y_valid)], 
             verbose=False)



predictions = my_model.predict(x_valid)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
predictions_df = pd.Series(predictions)

print("---------------------------")
print(y_valid)
print("---------------------------")
print(predictions_df)


plt.figure()
#predictions_df.plot(c='r')
y_valid.plot(c='r')
y_train.plot(c='b')
plt.show()

"""
df=pd.DataFrame(y_valid)
df.to_csv("output.csv")
"""