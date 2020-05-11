import pandas as pd
import numpy as np
import datetime as dt
import os 
from os import listdir
from os.path import isfile, join, isdir
from constants import macroWords, sectorWords, tickerList, companies
import re

def parseDate(str, format):
	return dt.datetime.strptime(str, format)

def extractNews():
	newsMicro = []
	newsMacro = []
	newsMisc = []

	startDate = parseDate('2006-10-20', '%Y-%m-%d')
	endDate = parseDate('2013-11-20', '%Y-%m-%d')
	currDate = startDate
	while currDate <= endDate:
		dateStr = currDate.strftime('%Y%m%d')
		pdir = 'Data/financial-news-dataset-master/ReutersNews106521/'+ dateStr
		if not isdir(pdir):
			currDate += dt.timedelta(days=1) 
			continue
		for file in listdir(pdir):
			if isfile(join(pdir, file)):
				f = open(join(pdir, file), 'r')	
				x = parseArticle(f.read())
				if x == None:
					print(join(pdir, file))
					continue
				newsType, data_tuple = (x[0], x[1])
				f.close()
				if newsType == 'micro':
					newsMicro.append(data_tuple)
				elif newsType == 'macro':
					newsMacro.append(data_tuple)
				else:
					newsMisc.append(data_tuple)

		currDate += dt.timedelta(days=1)  

	cols = ['date', 'tickers', 'title', 'headline', 'fulltext'] 
	df_micro = pd.DataFrame(newsMicro, columns = cols)  
	df_micro.to_csv('Data/stocks_newsMicro.csv', index=False)

	cols = ['date', 'industries', 'title', 'headline', 'fulltext'] 
	df_macro = pd.DataFrame(newsMacro, columns = cols)  
	df_macro.to_csv('Data/stocks_newsMacro.csv', index=False)

	cols = ['date', 'title', 'headline', 'fulltext'] 
	df_misc = pd.DataFrame(newsMisc, columns = cols)  
	df_misc.to_csv('Data/stocks_newsMisc.csv', index=False)

	# cleanup()

def parseArticle(text):
	text = text.lower()
	lines = text.split('-- ')
	try:
		if lines[0] == '':
			lines = lines[1:]
		title = lines[0].strip('\n')
	except:
		return None

	ind = -2
	success = False

	while not success and len(lines) + ind >= 0:
		try:
			date = parseDate(lines[ind][:-5], '%a %b %d, %Y %I:%M%p')
			success = True
		except ValueError:
			ind -= 1 

	if not success:
		return None

	rawtext = ''.join(lines[len(lines) + ind + 1:]).split('\n\n \n\n ')[-1]
	rawtext = re.split(r'\n\s*\n\s*', rawtext)
	try: 
		headline = re.sub('[^A-Za-z0-9 $.&-]+', '', rawtext[-2])
	except IndexError:
		headline = re.sub('[^A-Za-z0-9 $.&-]+', '', rawtext[-1])

	fulltext = re.sub('[^A-Za-z0-9 $.,&-]+', '', rawtext[-1])
	
	tickers = {x for x in tickerList if (" " +x+".") in headline}

	# Look for company name in headline
	if len(tickers) == 0:
		tickers = {companies[x] for x in companies.keys() if x in headline}

	if len(tickers) == 0:
 		if any(x in headline for x in macroWords):
 			industries =  {x for x in sectorWords if x in fulltext}
 			return 'macro', [date, industries, title, headline, fulltext] 
 		else:
 			return 'misc', [date, title, headline, fulltext]
	else :
	 	return 'micro', [date, tickers, title, headline, fulltext]


def processStockData():
	startDate = parseDate('2006-10-20', '%Y-%m-%d')
	endDate = parseDate('2013-11-21', '%Y-%m-%d')
	tickers = pd.read_csv('Data/constituents.csv')
	
	for symbol in tickers.get('Symbol'):
		ticker = symbol
		data = []
		fp = 'Data/stocks/' + ticker + '.csv'
		if isfile(fp) :
			f = open(fp, 'r')
			lineNumber = 0 
			for line in f:
				if lineNumber != 0:
					row = line.rstrip('\n').split(",")
					date = parseDate(row[0], '%Y-%m-%d')
					if date >= startDate and date <= endDate:
						print(row)
						data.append([date, row[1], row[2], row[3], row[4], row[5], row[6]])
				lineNumber+=1
			f.close()
			cols = ['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume']
			df = pd.DataFrame(data, columns = cols)  
			df.to_csv('Data/stocks_dateRange/' + ticker.lower() + '.csv', index=False)

		else:
			print("Error: File " + ticker + " does not appear to exist.")

def main():
    processStockData()
    # f = open('Data/financial-news-dataset-master/ReutersNews106521/20061213/businesspro-merck-vioxx-verdict-dc-idUSWAA00004520061213', 'r')
    # print(parseArticle(f.read()))
    # f.close()
    # extractNews()

if __name__ == "__main__":
    main()



