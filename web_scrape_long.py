import requests
import time
import random
import pandas as pd
from bs4  import BeautifulSoup


pageNumber = 300

titles = []
tickers = []
authors = []
date_times = []

HEADER = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"}
# driver = webdriver.Chrome()

# driver.get(url)

#while pageNumber <= 310:
url = "https://seekingalpha.com/stock-ideas/long-ideas?page=" + str(pageNumber)
try:
	r = requests.get(url,  headers = HEADER)
except Exception as e:
	print(e)

content = r.content
print(content)
soup = BeautifulSoup(content, features="html.parser")
for li in soup.findAll('li', {"class":"article media"}):
	call_dict = {}
	titles.append(li.find('a', {"class": "a-title"}).text)
	temp = li.find('div', {"class": "a-info"})
	a_list = temp.findAll('a')
	if a_list != None:
		tickers.append(a_list[0].text)
		if len(a_list) > 1:
			authors.append(a_list[-1].text)
		else:
			authors.append('')
		span_list = temp.findAll('span')
		if len(span_list) == 8:
			date_times.append(span_list[2].text)
		elif len(span_list) == 9:
			date_times.append(span_list[3].text)
		else:
			date_times.append('')
	# wait_review = WebDriverWait(driver, 5000)
	# driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
	# button = driver.find_element_by_xpath('//li[@class="next"]')
	# wait_review = WebDriverWait(driver, 10000)
	# button.click()
	# time.sleep(5)
	#pageNumber += 1

df = pd.DataFrame({'article_title':titles,'article_ticker':tickers,'article_author':authors, 'article_date_time': date_times}) 
df.to_csv('long_ratings.csv', index=False, encoding='utf-8')
#driver.close()


