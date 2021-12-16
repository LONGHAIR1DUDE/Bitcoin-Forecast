import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

import plotly.io as pio
pio.templates
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
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
file.write(str(int(predictions_df[0])))
file.close()
#Prophet prediciton Model
data=pd.read_csv("inputProphet.csv")
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
prediction_prophet=m.predict(futur)
print(prediction_prophet)
prediction=prediction_prophet.loc[prediction_prophet['ds'] == '2021-12-17']
print(prediction)
fileProphet = open("sampleProphet.txt", "w+")
fileProphet.write(str(int(prediction.yhat)))
fileProphet.close()





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
#<------------------------------LSTM-------------------------------->
# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]
 
# load dataset
series =read_csv('inputProphet.csv',index_col='Date',parse_dates=True,dayfirst=True)
series=series['Ouv.']
# split_date ='2020-01-31'
# series = series.loc[series.index>=split_date]
print(series.head())



 
# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
 
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

 
# split data into train and test-sets
train, test = supervised_values[0:-30], supervised_values[-30:]
 
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)
 
# fit the model
lstm_model = fit_lstm(train_scaled, 1, 10, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)
 
# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
 
# report performance
rmse = sqrt(mean_squared_error(raw_values[-30:], predictions))
print('Test RMSE: %.3f' % rmse)	
print(predictions)
fileR = open("sampleRmse.txt", "w+")
fileR.write(str(rmse))
fileR.close()

# line plot of observed vs predicted
 

pyplot.plot(raw_values[-30:])
pyplot.plot(predictions)
pyplot.show()

#<---------------------------Graphs--------------------------->
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
    x = prediction_prophet.ds,
    y= prediction_prophet.yhat,
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
    name = '10 WMA',
    marker_color='yellow'
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
filearima.write(str(int(data_ARIMA_month2.forecast.get("2021-12-31").drop_duplicates().array[0])))
filearima.close()

data = [trace1,trace2,trace3,trace4,trace5]
fig = go.Figure(data=data,layout=layout)
fig.update_layout(template="plotly_dark")
fig.write_html("output/output.html")
#<-----------------------Xg/bitcoin Graph ---------------------->
trace = go.Scatter(
    x = ax,
    y= bitcoin['Ouv.'],
    mode = 'lines+markers',
    name = 'Open'
)
traceXg = go.Scatter(
    x = ax,
    y= predictions_df,
    mode = 'lines',
    name = 'XgBoost Forecast'
)

dataXgboost = [trace,traceXg,trace5]
fig2 = go.Figure(data=dataXgboost,layout=layout)
fig2.update_layout(template="plotly_dark")
fig2.write_html("output/outputXg.html")
#<-----------------------Prophet/bitcoin Graph ---------------------->
trace = go.Scatter(
    x = ax,
    y= bitcoin['Ouv.'],
    mode = 'lines+markers',
    name = 'Open'
)
traceP = go.Scatter(
    x = prediction_prophet.ds,
    y= prediction_prophet.yhat,
    mode = 'lines',
    name = 'FaceBook Prophet Forecast'
)
dataP = [trace,traceP,trace5]
fig3 = go.Figure(data=dataP,layout=layout)
fig3.update_layout(template="plotly_dark")
fig3.write_html("output/outputProphet.html")
#<-----------------------Arima/bitcoin Graph ---------------------->
trace = go.Scatter(
    x = ax,
    y= bitcoin['Ouv.'],
    mode = 'lines+markers',
    name = 'Open'
)
traceA = go.Scatter(
    x = ax_ARIMA,
    y= data_ARIMA_month2.forecast,
    mode = 'lines',
    name = 'ARIMA Forecast',
    marker_color='lime'
)
dataA = [trace,traceA,trace5]
fig4 = go.Figure(data=dataA,layout=layout)
fig4.update_layout(template="plotly_dark")
fig4.write_html("output/outputARIMA.html")
print("<----------------------------------------->")
print("Output File is Ready")







