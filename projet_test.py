import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

import plotly.io as pio
pio.templates

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
import seaborn as sns; sns.set_theme()
from prophet import Prophet
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
import fbprophet as fbp
import warnings;
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')

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
    fileXg = open("sampleMae.txt", "w+")
    fileXg.write(str(mean_absolute_error(predictions, y_valid)))
    fileXg.close()
    return pd.Series(predictions)


# Inverse Box-Cox Transformation Function
def invboxcox(y,lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))


bitcoin = pd.read_csv('BTC-USD.csv',index_col='Date',parse_dates=[0],dayfirst=True)
#cols_to_use = ['Dernier','Plus Haut','Plus Bas','Vol.','Variation %']



formating('Ouv.',bitcoin,'prix')
formating('Dernier',bitcoin,'prix')
formating('Plus Haut',bitcoin,'prix')
formating('Plus Bas',bitcoin,'prix')
formating('Vol.',bitcoin,'volume')
formating('Variation %',bitcoin,'variation')



df=pd.DataFrame(bitcoin)
df.to_csv("inputProphet.csv")




#Data Splitting
split_date = '2021-06-01'
data_train = bitcoin.loc[bitcoin.index <= split_date].copy()
data_test = bitcoin.loc[bitcoin.index > split_date].copy()
date_listD = data_test.index.to_frame(index=False)

print("<---------------------------THIS IS DATA_TEST------------------------------------->")
print(date_listD)
#Xgboost model prediction
X_train, y_train = create_features(data_train, label='Ouv.')
X_test, y_test = create_features(data_test, label='Ouv.')


predictions_df = XGB_model(X_train,y_train,X_test,y_test)
print("<---------------------------THIS IS 26/11/2021 pred------------------------------------->")
print(predictions_df[0])
file = open("sample.txt", "w+")
file.write(predictions_df[0].astype('str'))
file.close()
#Prophet prediciton Model
data=bitcoin
data['Date']=pd.DatetimeIndex(data['Date'],dayfirst=True)

print(data)




data['ds']=data['Date']
data['y']=data['Ouv.']
data.drop(['Dernier','Plus Haut','Plus Bas','Vol.','Variation %','Date','Ouv.'],axis=1,inplace=True)



print(data.dtypes)
print(data.head())




m=fbp.Prophet(interval_width=0.95,daily_seasonality=True)
model=m.fit(data)

futur=m.make_future_dataframe(periods=300,freq='D')
prediction=m.predict(futur)
# prediction=prediction[['ds','yhat']]






#Seaborn data correlation heatmap
df=bitcoin.astype(float)
#sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0)

plt.show()
##<-----------------------------ARIMA--------------------------------------->
#ARIMA model data pre-processing

data_ARIMA =bitcoin.resample('D').mean()
data_ARIMA_month = data_ARIMA.resample('M').mean()
data_ARIMA_year = data_ARIMA.resample('A-DEC').mean()
data_ARIMA_Quarter = data_ARIMA.resample('Q-DEC').mean()
# PLOTS
fig = plt.figure(figsize=[15, 7])
plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)

plt.subplot(221)
plt.plot(data_ARIMA["Ouv."], '-', label='By Days')
plt.legend()

plt.subplot(222)
plt.plot(data_ARIMA_month["Ouv."], '-', label='By Months')
plt.legend()

plt.subplot(223)
plt.plot(data_ARIMA_Quarter["Ouv."], '-', label='By Quarters')
plt.legend()

plt.subplot(224)
plt.plot(data_ARIMA_year["Ouv."], '-', label='By Years')
plt.legend()

# plt.tight_layout()
plt.savefig("output/bitcoin_d_m_y_q.png")
#Stationarity check and STL-decomposition of the series

plt.figure(figsize=[15,7])
sm.tsa.seasonal_decompose(data_ARIMA_month["Ouv."]).plot()
print("Seasonal Decomp Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(data_ARIMA_month["Ouv."])[1])
plt.savefig("output/seasonal_decomp.png")
#Box-Cox Transformations
data_ARIMA_month['Ouv_box'], lmbda = stats.boxcox(data_ARIMA_month["Ouv."])
print("Box-Cox Trans Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(data_ARIMA_month["Ouv."])[1])
#Seasonal differentiation

data_ARIMA_month['Ouv_box_diff'] = data_ARIMA_month['Ouv_box'] - data_ARIMA_month['Ouv_box'].shift(12)
print("Seasonal Diff Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(data_ARIMA_month['Ouv_box_diff'][12:])[1])


# Regular differentiation
data_ARIMA_month['Ouv_box_diff2'] = data_ARIMA_month['Ouv_box_diff'] - data_ARIMA_month['Ouv_box_diff'].shift(1)
plt.figure(figsize=(15,7))

# STL-decomposition
sm.tsa.seasonal_decompose(data_ARIMA_month['Ouv_box_diff2'][13:]).plot()   
print("STL Decomp Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(data_ARIMA_month['Ouv_box_diff2'][13:])[1])

plt.savefig("output/STL_decomp.png")
# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
plt.figure(figsize=(15,7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(data_ARIMA_month['Ouv_box_diff2'][13:].values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(data_ARIMA_month['Ouv_box_diff2'][13:].values.squeeze(), lags=48, ax=ax)
plt.tight_layout()
plt.savefig("output/ARIMA_corr.png")
# Initial approximation of parameters
Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D=1
d=1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model=sm.tsa.statespace.SARIMAX(data_ARIMA_month['Ouv_box'], order=(param[0], d, param[1]), 
                                        seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
    except ValueError:
        print('wrong parameters:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
    print("Done")


# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())
#Residue Analysis
# STL-decomposition
plt.figure(figsize=(15,7))
plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Residuals')
ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)

print("Dickey–Fuller test:: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])

plt.tight_layout()
plt.savefig("output/STL_resid.png")
# Prediction
data_ARIMA_month2 = data_ARIMA_month[['Ouv.']]
date_list = [datetime(2021,12,31),datetime(2022,1,31),datetime(2022,2,28),datetime(2022,3,31),datetime(2022,4,30)]
future = pd.DataFrame(index=date_list, columns= data_ARIMA_month.columns)
data_ARIMA_month2 = pd.concat([data_ARIMA_month2, future])

data_ARIMA_month2['forecast'] = invboxcox(best_model.predict(start=0, end=122), lmbda)
data_ARIMA_month2['forecast'] = data_ARIMA_month2['forecast'].drop_duplicates() 
plt.figure(figsize=(15,7))
data_ARIMA_month2["Ouv."].plot()
data_ARIMA_month2.forecast.drop_duplicates().plot(color='r', ls='--', label='Predicted Ouv')
print(data_ARIMA_month2.forecast)
plt.legend()
plt.title('Bitcoin exchanges, by months')
plt.ylabel('mean USD')
plt.savefig("output/ARIMA_forecast.png")

##<-----------------------------ARIMA DAILY--------------------------------------->
##ARIMA model data pre-processing
#
#data_ARIMA =bitcoin.resample('D').mean()
#data_ARIMA_month = data_ARIMA.resample('M').mean()
#data_ARIMA_year = data_ARIMA.resample('A-DEC').mean()
#data_ARIMA_Quarter = data_ARIMA.resample('Q-DEC').mean()
#data_ARIMA['Ouv.'].fillna(method='ffill', inplace=True)
##Debbuging step only : Outputs a csv to review data after the fact
#df=pd.DataFrame(data_ARIMA["Ouv."])
#df.to_csv("output.csv")
## PLOTS
#fig = plt.figure(figsize=[15, 7])
#plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)
#
#plt.subplot(221)
#plt.plot(data_ARIMA["Ouv."], '-', label='By Days')
#plt.legend()
#
#plt.subplot(222)
#plt.plot(data_ARIMA_month["Ouv."], '-', label='By Months')
#plt.legend()
#
#plt.subplot(223)
#plt.plot(data_ARIMA_Quarter["Ouv."], '-', label='By Quarters')
#plt.legend()
#
#plt.subplot(224)
#plt.plot(data_ARIMA_year["Ouv."], '-', label='By Years')
#plt.legend()
#
## plt.tight_layout()
#plt.savefig("output/bitcoin_d_m_y_q.png")
##Stationarity check and STL-decomposition of the series
#
#plt.figure(figsize=[15,7])
#sm.tsa.seasonal_decompose(data_ARIMA["Ouv."]).plot()
#print("Seasonal Decomp Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(data_ARIMA["Ouv."])[1])
#plt.savefig("output/seasonal_decomp.png")
##Box-Cox Transformations
#data_ARIMA['Ouv_box'], lmbda = stats.boxcox(data_ARIMA["Ouv."])
#print("Box-Cox Trans Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(data_ARIMA["Ouv."])[1])
##Seasonal differentiation
#print(data_ARIMA['Ouv_box'])
#data_ARIMA['Ouv_box_diff'] = data_ARIMA['Ouv_box'] - data_ARIMA['Ouv_box'].shift(12)
#print("Seasonal Diff Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(data_ARIMA['Ouv_box_diff'][12:])[1])
#
#
## Regular differentiation
#data_ARIMA['Ouv_box_diff2'] = data_ARIMA['Ouv_box_diff'] - data_ARIMA['Ouv_box_diff'].shift(1)
#plt.figure(figsize=(15,7))
#
## STL-decomposition
#sm.tsa.seasonal_decompose(data_ARIMA['Ouv_box_diff2'][13:]).plot()   
#print("STL Decomp Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(data_ARIMA['Ouv_box_diff2'][13:])[1])
#
#plt.savefig("output/STL_decomp.png")
## Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
#plt.figure(figsize=(15,7))
#ax = plt.subplot(211)
#sm.graphics.tsa.plot_acf(data_ARIMA['Ouv_box_diff2'][13:].values.squeeze(), lags=48, ax=ax)
#ax = plt.subplot(212)
#sm.graphics.tsa.plot_pacf(data_ARIMA['Ouv_box_diff2'][13:].values.squeeze(), lags=48, ax=ax)
#plt.tight_layout()
#plt.savefig("output/ARIMA_corr.png")
## Initial approximation of parameters
#Qs = range(0, 2)
#qs = range(0, 2)
#Ps = range(0, 2)
#ps = range(0, 2)
#D=1
#d=1
#parameters = product(ps, qs, Ps, Qs)
#parameters_list = list(parameters)
#len(parameters_list)
#
## Model Selection
#results = []
#best_aic = float("inf")
#warnings.filterwarnings('ignore')
#for param in parameters_list:
#    try:
#        model=sm.tsa.statespace.SARIMAX(data_ARIMA['Ouv_box'], order=(param[0], d, param[1]), 
#                                        seasonal_order=(param[2], D, param[3], )).fit(disp=-1)
#    except ValueError:
#        print('wrong parameters:', param)
#        continue
#    aic = model.aic
#    if aic < best_aic:
#        best_model = model
#        best_aic = aic
#        best_param = param
#    results.append([param, model.aic])
#    print("Done")
#
#
## Best Models
#result_table = pd.DataFrame(results)
#result_table.columns = ['parameters', 'aic']
#print(result_table.sort_values(by = 'aic', ascending=True).head())
#print(best_model.summary())
##Residue Analysis
## STL-decomposition
#plt.figure(figsize=(15,7))
#plt.subplot(211)
#best_model.resid[13:].plot()
#plt.ylabel(u'Residuals')
#ax = plt.subplot(212)
#sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)
#
#print("Dickey–Fuller test:: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])
#
#plt.tight_layout()
#plt.savefig("output/STL_resid.png")
## Prediction
#data_ARIMA_month2 = data_ARIMA[['Ouv.']]
#
#date_list = ["2021-06-01","2021-06-02","2021-06-03","2021-06-04","2021-06-05"]
#future = pd.DataFrame(index=date_listD['Date'], columns= data_ARIMA.columns)
#data_ARIMA_month2 = pd.concat([data_ARIMA, future])
#
#data_ARIMA_month2['forecast'] = invboxcox(best_model.predict(start=3400, end=3800
#                                                             ), lmbda)
#data_ARIMA_month2['forecast'] =data_ARIMA_month2['forecast'].drop_duplicates()
#plt.figure(figsize=(15,7))
#data_ARIMA_month2["Ouv."].plot()
#data_ARIMA_month2.forecast.plot(color='r', ls='--', label='Predicted Ouv')
#print(data_ARIMA_month2.forecast)
#plt.legend()
#plt.title('Bitcoin exchanges, by months')
#plt.ylabel('mean USD')
#plt.savefig("output/ARIMA_forecast.png")
weights = np.arange(1,11)
wma10 = bitcoin['Ouv.'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
bitcoin['Our 10-day WMA'] = np.round(wma10, decimals=3)
#Model Comparision Graphs
ax = list(bitcoin.index)
ax_ARIMA = list(data_ARIMA_month2.index)
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
    name = 'XgBoost Forecast'
)
trace3 = go.Scatter(
    x = ax,
    y= predicition,
    mode = 'lines',
    name = 'FaceBook Prophet Forecast'
)
trace4 = go.Scatter(
    x = ax_ARIMA,
    y= data_ARIMA_month2.forecast,
    mode = 'lines',
    name = 'ARIMA Forecast'
)
trace5 = go.Scatter(
    x = ax,
    y= bitcoin['Our 10-day WMA'],
    mode = 'lines',
    name = '10 WMA'
)
layout = dict(
    title='Historical Bitcoin Open Prices (2012-2021) and Model Predictions',
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
            ]),
             bgcolor="black"
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)
print('================================ARIMA 26-11 pred ========================================')


filearima = open("sampleARIMA.txt", "w+")
filearima.write(str(data_ARIMA_month2.forecast.get("2021-12-31").drop_duplicates().array[0]))
filearima.close()

data = [trace1,trace2,trace3,trace4,trace5]
fig = go.Figure(data=data,layout=layout)
fig.update_layout(template="plotly_dark")
fig.write_html("output/output.html")

dataXgboost = [trace1,trace2]
fig = go.Figure(data=dataXgboost,layout=layout)
fig.update_layout(template="plotly_dark")
fig.write_html("output/outputXg.html")

print("<----------------------------------------->")
print("Output File is Ready")




model = Sequential()
model.add(LSTM(100, input_shape=(x_train.shape[0], x_train.shape[1])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# fit network
#history = model.fit(x_train, y_train, epochs=20, batch_size=70, validation_data=(x_valid, y_valid), verbose=2, shuffle=False)

