import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


def convertion(tab):
	temp=tab.replace('.','',1)
	temp=temp.replace(',','.',1)

	return float(temp)


bitcoin=pd.read_csv('BTC-USD.csv',index_col='Date',parse_dates=True)
print(bitcoin.head())
print(bitcoin.shape)
print(type(bitcoin))
print(bitcoin['Ouv.'])

for i in range(bitcoin['Ouv.'].shape[0]):
	bitcoin['Ouv.'][i]=convertion(bitcoin['Ouv.'][i])


print(bitcoin.shape)
print(type(bitcoin))
print(bitcoin['Ouv.'])

bitcoin['2020']['Ouv.'].plot(figsize=(15,6))
plt.show()