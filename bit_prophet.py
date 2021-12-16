import warnings;
warnings.simplefilter('ignore')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import fbprophet as fbp


data=pd.read_csv("inputProphet.csv")
data['Date']=pd.DatetimeIndex(data['Date'],dayfirst=True)

print(data)




# for i in range(len(data)):
# 	data.iloc[i,0]=str(data.iloc[i,0])
# 	s=data.iloc[i,0]
# 	year=s[-4:]
# 	month=s[-6:-4]
# 	day=s[0:-6]
# 	data.iloc[i,0]=day+'-'+month+'-'+year




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
print(prediction.head())

plot1 = m.plot(prediction)

plt.show()