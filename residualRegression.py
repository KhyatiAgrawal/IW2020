# Code partially adapted from
# Citation
# https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as sma
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
from constants import macroWords, sectorWords, tickerList, companies
import warnings
from os.path import isfile, join, isdir
from os import mkdir
import sys
import pickle
from pandas import Timestamp
import time

def readNews(ticker):
	# Read the sentiment Scores data
	fp = 'news_scores/'
	df_news = pd.read_csv(fp + ticker + ".csv", parse_dates=['datetime'], index_col=['datetime'])
	df_news = df_news.rename(index={'datetime': 'date'})
	return df_news

def residualRegression(ticker, savePath):
	# Load model
	sp = 'results/' + ticker + "/"
	with open(sp + "model.pkl", 'rb') as pickle_file:
		model = pickle.load(pickle_file)

	# Read data
	fp = 'Data/stocks_dateRange/'
	data = pd.read_csv(fp + ticker + ".csv", parse_dates=['date'], index_col=['date'])
	data = data['adjclose']

	# Apply model 1
	predictions = model.apply(data)
	pred = predictions.get_prediction(dynamic=False)
	y_forecasted = pred.predicted_mean
	df_residuals = data - y_forecasted
	df_residuals = df_residuals.loc[((df_residuals <= 2) & (df_residuals >= -2))]

	df_news = readNews(ticker)

	# Combine the dataframes
	df = df_residuals.to_frame().join(df_news, how='outer').iloc[1:]
	df = df.dropna()
	df = df.rename(columns ={0: 'y'})

	closestDate = df.index[df.index >= Timestamp('2013-01-01 00:00:00')][0]	
	df_train = df[:closestDate]
	df_test = df[closestDate:]
	print(df_train.shape)
	print(df_test.shape)


	string = ""
	for col in list(df_news.columns):
		string += "+ " + col 
	fmula = "y ~ " + string[2:]
	result = sma.ols(formula=fmula, data=df_train).fit()
	print(result.summary())
	result.save(savePath +"model_part2.pkl")

	# Calculate the MSE on training data
	yhat = result.predict()
	mse_train = ((df_train['y'] - yhat)**2).mean()
	print('The MSE on training set is {}'.format(round(mse_train, 4)))

	# Create the predicted vs actual plot
	fig, ax = plt.subplots()
	ax.scatter(df_train['y'], yhat, s=5)
	ax.plot([df_train['y'].min(), df_train['y'].max()], [df_train['y'].min(), df_train['y'].max()], lw=1)
	ax.set_ylabel("Predicted shock")
	ax.set_xlabel("Actual shock")
	ax.set_title("Predicting shocks for " + ticker)
	plt.savefig(savePath + "predVSActualResidual.png")
	plt.close()


	# Calculate the MSE on the test data
	print('First out of sample date is ')
	print(closestDate)
	yhat = result.predict(df_test.iloc[:,1:])
	mse_test = ((df_test['y'] - yhat)**2).mean()
	print('The MSE on test set is {}'.format(round(mse_test, 4)))

	return model


tickers = {
'wmt': 'wal-mart stores',
'f': 'ford motor', 
'bac': 'bank of america corp', 
'low': 'lowes cos.', 
'aapl': 'gartner inc', 
}

for key, value in tickers.items():
	sp = 'results/' + key + "/"
	orig_stdout = sys.stdout
	f = open(sp+'out_part2.txt', 'w')
	sys.stdout = f

	residualRegression(key, sp)

	sys.stdout = orig_stdout
	f.close()




