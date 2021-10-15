import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


import scipy as sp

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


bitcoin = pd.read_csv('BTC-USD.csv',index_col='Date',parse_dates=[0],dayfirst=True)
cols_to_use = ['Dernier','Plus Haut','Plus Bas','Vol.','Variation %']



formating('Ouv.',bitcoin,'prix')
formating('Dernier',bitcoin,'prix')
formating('Plus Haut',bitcoin,'prix')
formating('Plus Bas',bitcoin,'prix')
formating('Vol.',bitcoin,'volume')
formating('Variation %',bitcoin,'variation')




X = bitcoin[cols_to_use]
# Select target
y = bitcoin['Ouv.']


#x_train = X['2012':'2020']
#x_valid = X['2021']
#
#y_train=y['2012':'2020']
#y_valid=y['2021']

split_date = '2021-05-01'
data_train = bitcoin.loc[bitcoin.index <= split_date].copy()
data_test = bitcoin.loc[bitcoin.index > split_date].copy()
def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['Day'] = df['date'].dt.day
    
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['Vol.'] = df['Vol.'].astype(float)
    df['Plus Haut'] = df['Plus Haut'].astype(float)
    df['Dernier'] = df['Dernier'].astype(float)
    df['Plus Bas'] = df['Plus Bas'].astype(float)
    df['Variation %'] = df['Variation %'].astype(float)
    X = df[['Day','quarter','month','year','Vol.','Dernier','Plus Haut','Plus Bas','Variation %']]
    if label:
        y = df[label]
        return X, y
    return X

X_train, y_train = create_features(data_train, label='Ouv.')
X_test, y_test = create_features(data_test, label='Ouv.')


df=pd.DataFrame(X_train)
df.to_csv("output.csv")
def XGB_model(x_train, y_train,x_valid,y_valid):
    my_model = XGBRegressor(n_estimators=3000, learning_rate=0.01,tree_method='hist',max_bin=300,objective ='reg:squarederror',
         alpha = 6)
    my_model.fit(x_train, y_train, 
         early_stopping_rounds=5, 
         eval_set=[(x_train,y_train),(x_valid, y_valid)], 
         verbose=False)
    predictions = my_model.predict(x_valid)
    print("<----------------------------------------->")
   
    print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
    return pd.Series(predictions)


predictions_df = XGB_model(X_train,y_train,X_test,y_test)




ax = list(bitcoin.index)
trace1 = go.Scatter(
    x = ax,
    y= bitcoin['Ouv.'],
    mode = 'lines+markers',
    name = 'Open'
)
trace2 = go.Scatter(
    x = ax,
    y= predictions_df,
    mode = 'lines',
    name = 'XgBoost_pred_Open'
)
trace3 = go.Scatter(
    x = ax,
    y= bitcoin['Plus Haut'],
    mode = 'lines',
    name = 'highest'
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
data = [trace1,trace2,trace3]
fig = go.Figure(data=data,layout=layout)
#iplot(fig, filename = "Time Series with Rangeslider") 
print("<----------------------------------------->")
fig.write_html("output.html")
print("Output File is Ready")
df=bitcoin.astype(float)
print(df.corr())
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0)