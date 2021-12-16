import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf


def time_series_model(neurons='64',optimizer='adam',model_save_path="model",loss="mae") :
	model = tf.keras.models.Sequential([
	tf.keras.layers.LSTM(neurons),
	tf.keras.layers.Dense(1)])


	model.compile(optimizer=optimizer,loss=loss,metrics=[loss])
	callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
	monitor=loss,
	save_weights_only=True,
	save_best_only=True,
	mode='auto')]

	return model,callbacks



def multivariate_data(dataset,target,start_index,end_index,history_size,target_size) :
	data=[]
	labels=[]

	start_index=start_index+history_size
	if end_index is None :
		end_index=len(dataset)-target_size

	for i in range(start_index,end_index):
		indices=list(range(i-history_size,i))
		temp=[]
		for j in range(len(indices)):
			temp.append(dataset[j])
		data.append(temp)
		labels.append(target[i+target_size])

	return np.array(data),np.array(labels)


#########################################################################

df = pd.read_csv("output.csv")
dataset_1 = df['Ouv.']
dataset=list(reversed(dataset_1))


days_to_predict=50
history_points=50
train_size=len(dataset)-(history_points+days_to_predict)
print("Training size is {}".format(train_size))
print("Test size is {}".format(len(dataset)-train_size))

test=dataset[train_size : train_size+ history_points]
to_predict=list(dataset)[-1]


x_train,y_train =multivariate_data(dataset=dataset,
				target=dataset,
				start_index=0,
				end_index=train_size,
				history_size=history_points,
				target_size=days_to_predict )

print("Affichage de la shape de x_train et y_train avant modif dim")	
print(x_train.shape,y_train.shape)

x_test=np.array(test)
if len(x_train.shape)==2:
	x_train=np.expand_dims(x_train,axis=-1)


print("Affichage de la shape de x_train et y_train apres modif dim")
print(x_train.shape,y_train.shape)
x_test=np.expand_dims(np.expand_dims(x_test,axis=-1),axis=0)
print("Affichage de la shape de x_test")
print(x_test.shape)


point_model,callbacks= time_series_model(neurons=64,
	optimizer='adam',
	model_save_path='model',
	loss="mse")

history = point_model.fit(x_train,y_train,epochs=10,batch_size=16,verbose=0,shuffle=False,callbacks=callbacks)

point_model.load_weights('model')
ypred = point_model.predict(x_test)
pred=ypred[0]


fig=go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(dataset))),y=list(dataset)
				,mode='lines',name='True'))

fig.add_trace(go.Scatter(x=list(range(len(dataset) - (days_to_predict+len(test)),
	len(dataset)-days_to_predict)),
	y=test,
	mode='lines+markers',name='Test'))

fig.add_trace(go.Scatter(x=[len(dataset)-1],
	y=[to_predict],mode="lines+markers",name='To predict'))

fig.add_trace(go.Scatter(x=[len(dataset)-1],
	y=pred,mode="lines+markers",name='Prediction'))

fig.update_layout(
	title="Prediction réseau de neurones",
	xaxis_title="Days",
	yaxis_title="Prix Bitcoin",
	font=dict(
		family="Courier New, monospace",
		size=18,
		color="#7f7f7f"
		)

	)

fig.show()