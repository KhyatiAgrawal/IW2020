# Code partially adapted from
# Citation
# https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
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

def readData(path, ticker, stockname, timeSeriesCol):
	print("Reading data for " + stockname + " " + ticker)
	df = pd.read_csv(path + ticker + ".csv", parse_dates=['date'], index_col=['date'])
	df_col = df[timeSeriesCol]
	return df_col

def testTrainSplit(df):
	df_train = df[0:int(0.8*df.shape[0])]
	df_test = df[int(0.8*df.shape[0]):]
	return df_train, df_test

def predictAndPlot(data, startDate, model, savePath, i, train = True):
	predictions = model.apply(data)
	pred = predictions.get_prediction(start=startDate, dynamic=False)
	plt.style.use('seaborn-poster')
	
	if train:
		plt.figure(i)
		ax = data.plot(label='Observed')
		pred_ci = pred.conf_int()
		ax.fill_between(pred_ci.index,
             pred_ci.iloc[:, 0],
             pred_ci.iloc[:, 1], color='k', alpha=.2)
		pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.6)
	else:
		plt.figure(i+1)
		ax2 = data[startDate:].plot(label='Observed')
		pred.predicted_mean.plot(ax=ax2, label='One-step ahead Forecast', alpha=.6)
	
	plt.legend()
	
	y_forecasted = pred.predicted_mean
	y_truth = data[startDate:]
	mse = ((y_forecasted - y_truth) ** 2).mean()

	if not train:
		plt.savefig(savePath + "Predictions_test.png")
		print('The Mean Squared Test Error of our forecasts is {}'.format(round(mse, 2)))
	else:
		plt.savefig(savePath + "Predictions_training.png")
		print('The Mean Squared Training Error of our forecasts is {}'.format(round(mse, 2)))


def fitSARIMAX(pdq, pdqs, series, savePath,i, model = None):
	warnings.filterwarnings("ignore") 
	df_train, df_test = testTrainSplit(series)

	if model != None:
		predictAndPlot(df_train, df_train.index[100], model, savePath, i)
		predictAndPlot(series, df_test.index[0], model, savePath, i, False)
		return

	min_aic = sys.float_info.max
	for param in pdq:
	    for param_seasonal in pdqs:
	        try:
	            mod = sm.tsa.statespace.SARIMAX(df_train,
	                                            order=param,
	                                            seasonal_order=param_seasonal,
	                                            enforce_stationarity=True,
	                                            enforce_invertibility=False)

	            results = mod.fit(disp=0)
	            if (results.aic < min_aic):
	                min_aic = results.aic
	                best = (param, param_seasonal)
	        except:
	            continue

	mod = sm.tsa.statespace.SARIMAX(df_train,
	                                order=best[0],
	                                seasonal_order=best[1],
	                                enforce_stationarity=True,
	                                enforce_invertibility=False)

	results = mod.fit(disp=0)

	print(results.summary())
	results.save(savePath +"model.pkl")

	plt.style.use('seaborn-whitegrid')
	results.plot_diagnostics(figsize=(15, 12))
	plt.savefig(savePath + "residualPlots.png")
	predictAndPlot(df_train, df_train.index[100], results, savePath, i)
	predictAndPlot(series,df_test.index[0], results, savePath, i, False)
	return results

def main():
	warnings.filterwarnings("ignore") 
	fp = 'Data/stocks_dateRange/'
	#tickerDict = {v: k for k, v in companies.items()}
	tickerDict = {'v': 'visa inc.'}
	i = 0
	for key, value in tickerDict.items():

		# Create directory
		sp = 'results/' + key + "/"

		if not isdir(sp):
			try:
				mkdir(sp)
			except OSError:
				print ("Creation of the directory %s failed" % path)

		orig_stdout = sys.stdout
		f = open(sp+'out.txt', 'w')
		sys.stdout = f

		series = readData(fp, key, value, 'adjclose')
		print(series.head())

		p = q = range(0, 2)
		d = [1]
		pdq = list(itertools.product(p, d, q))

		period = [5]
		pdqs = list(itertools.product(p, d, q, period))

		if not isfile(sp + "model.pkl"):
			fitSARIMAX(pdq, pdqs, series, sp, i)
		else:
			with open(sp + "model.pkl", 'rb') as pickle_file:
				model = pickle.load(pickle_file)
			print(model.summary())
			fitSARIMAX(pdq, pdqs, series, sp, i, model)

		sys.stdout = orig_stdout
		f.close()
		i+=2

if __name__ == "__main__":
    main()

		













