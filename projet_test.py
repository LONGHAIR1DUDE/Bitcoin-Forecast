import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


import scipy as sp
import keras
import xgboost as xgb
import sklearn as sk
import plotly
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn import preprocessing
from scipy import optimize 
from plotly import tools
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import gc

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px

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


bitcoin = pd.read_csv('BTC-USD.csv',index_col='Date',parse_dates=True,dayfirst=True)
#cols_to_use = ['Dernier','Plus Haut','Plus Bas','Vol.','Variation %']
cols_to_use = ['Dernier','Plus Haut','Plus Bas','Vol.','Variation %']



formating('Ouv.',bitcoin,'prix')
formating('Dernier',bitcoin,'prix')
formating('Plus Haut',bitcoin,'prix')
formating('Plus Bas',bitcoin,'prix')
formating('Vol.',bitcoin,'volume')
formating('Variation %',bitcoin,'variation')



for i in cols_to_use:
    bitcoin[i] = bitcoin[i].astype(float)

#print(bitcoin)
#print(bitcoin.dtypes)
X = bitcoin[cols_to_use]
# Select target
y = bitcoin['Ouv.']



split_date ='2020-01-31'

x_train = X.loc[X.index<=split_date]
x_valid = X.loc['2021':]

y_train = y.loc[y.index<=split_date]
y_valid = y.loc['2021':]




print(x_train.shape[0])
print("-----------xtrain----------------")
print(x_train)
print("------------ytrain---------------")
print(y_train)
print("------------x_valid---------------")
print(x_valid)
print("-------------y_valid--------------")
print(y_valid)



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(x_train, y_train, 
             early_stopping_rounds=10, 
             eval_set=[(x_valid, y_valid)], 
             verbose=False)



predictions = my_model.predict(x_valid)
print("<----------------------------------------->")
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
predictions_df = pd.Series(predictions)
print("<----------------------------------------->")
print("---------------------------")
print(y_valid)
print("---------------------------")
print(predictions_df)




df=pd.DataFrame(bitcoin)
df.to_csv("output.csv")

ax = list(bitcoin.index)
trace1 = go.Scatter(
    x = ax,
    y= bitcoin['Ouv.'],
    mode = 'lines+markers',
    name = 'Ouv.'
)
trace2 = go.Scatter(
    x = ax,
    y= predictions_df,
    mode = 'lines',
    name = 'Xg_pred.'
)
layout = dict(
    title='Historical Bitcoin Prices (2012-2021) with the Slider ',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                #change the count to desired amount of months.
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=12,
                     label='1y',
                     step='month',
                     stepmode='backward'),
                dict(count=36,
                     label='3y',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)
data = [trace1,trace2]
fig = go.Figure(data=data,layout=layout)
iplot(fig, filename = "Time Series with Rangeslider") 
print("<----------------------------------------->")
fig.write_html("output.html")
print("Output File is Ready")




"""
model = Sequential()
model.add(LSTM(100, input_shape=(x_train.shape[0], x_train.shape[1])))
model.add(Dropout(0.2))
#    model.add(LSTM(70))
#    model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
"""
# fit network
#history = model.fit(x_train, y_train, epochs=20, batch_size=70, validation_data=(x_valid, y_valid), verbose=2, shuffle=False)

