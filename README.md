# [Bitcoin](https://coinmarketcap.com/currencies/bitcoin/) Price Forecast
### LIFPROJET RC3

 #### _Projet dans le cadre de L'UE [LIFPROJET](http://cazabetremy.fr/wiki/doku.php?id=projet:presentation) basée sur le sujet RC3 : Kaggle Challenges_


***
### **Objectif :**
###  Application de plusieurs modèles de prédictions issus de différentes Bibliothèques Python implementant des techniques de Deep et de Machine Learning sur les données historiques OHLCV (2012-2021) du [Bitcoin](https://coinmarketcap.com/currencies/bitcoin/) pour tenter de prédire le cours de celui-ci et présentation des résultats sous forme d'un site web (`index.html` ).
### Les modèles de prédiction implémentés :
 * XGBOOST  : eXtreme Gradient Boosting .
 * Facebook Prophet .
 * LSTM : Long short-term Memory.
 * ARIMA : Auto Regressive Intergated Moving Average .
---
### **Installation :** 
#### _On recommende l'utilisation de l'application dans un environnement anaconda fonctionnel._
* ### **Linux :**
  * #### On peut faire   `git clone` directement du page [GitLab](https://forge.univ-lyon1.fr/p1803192/lifprojet-rc3) du projet .
  * #### Pour exécuter l'application on se met dans le répertoire `~lifprojet-RC3` et lancer la commande : 
    `python3 projet_test.py`  
  * ####  Pour  mieux visualiser les données lancer la page web `index.html` dans votre navigateur web .
    #
    #### **Dépendances :**
  * ##### Numpy.
  * ##### Matplotlib.
  * ##### Pandas .
  * ##### plotly.
  * ##### Sklearn.
  * ##### Keras.
  * ##### Xgboost.
  * ##### Seaborn .
  * ##### Scipy.
  * ##### Statsmodels .
  * ##### Fbprophet .
---
### Contenu Du répertoire Git :
* `./output` : les fichiers des résultats géneré par `projet_test.py` .
* `./ouput/txt` : Les prédictions de chaque modèle  sur le dernier prix de Bitcoin aujourd'hui  et les MAE et RMSE de ces modèles.
* `./output/html` :  Contient les graphes des prédictions génerés par `projet_test.py` en format HTML.
* `./output/png` :  Contient quelques graphes des prédictions génerés par `projet_test.py` en format PNG.
* `index.html` : Fichier html de la page web pour visualiser les résultats.
* `projet_test.py` : Un fichier python qui contient l'implémentation de tous les modèles et génere des graphes et des résultats qu'on présente dans `index.html` .
------------------------
### Code : 
##### Les prédictions se font à l'aide des méthodes suivantes :
* Xgboost :
   ``` 
   def XGB_model(x_train, y_train,x_valid,y_valid):
    my_model = XGBRegressor(n_estimators=3000, learning_rate=0.01,tree_method='hist',objective ='reg:squarederror',
         alpha = 6)       #Je crée mon modèle à l'aide de la fonction XGBRegressor()
    my_model.fit(x_train, y_train, 
         early_stopping_rounds=5, 
         eval_set=[(x_train,y_train),(x_valid, y_valid)], 
         verbose=False)   #Je fait un fit() du modèle sur les données d'entrainement 
    predictions = my_model.predict(x_valid) #Je fait des prédiction a l'aide de predict() sur les données de validation (ou test).
    ```
    * `n_estimators` (int) – nombres d'arbre de boosting .
    * `learning_rate` (float) - le taux d'apprentissage une valeur plus bas introduit moins d'Overfitting
    * `tree_method` (str) - la méthode d'arbre à utiliser .
    * `objective` (str) - la tache ou la fonction d'apprentisage .
* LSTM :
  ```
  def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)#ici le fit() de notre modèle .
		model.reset_states()
	return model
  ```
  * `batch_size` (int) - nombre d'échantillon par évolution de gradient .
  ```
  def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size) #on fait des prédictions à l'aide de predict() sur X :jeu de donnée de validation
	return yhat[0,0]
  ```
* Facebook Prophet :
  ```
  m=fbp.Prophet(interval_width=0.95,daily_seasonality=True)
  model=m.fit(data)

  futur=m.make_future_dataframe(periods=300,freq='D')
  prediction_prophet=m.predict(futur)
  ```
  * `interval_width` (float) - intervalle d'incertitude (d'erreur)
  * `daily_seasonality` (bool) - spécifie la périodicité .


----------------------------------------------------------------
### Résultat :
![Sample](sample.png)
