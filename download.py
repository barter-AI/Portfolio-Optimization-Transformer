from datetime import datetime
from concurrent import futures

import pandas as pd
from pandas import DataFrame
# import pandas_datareader.data as web
import yfinance as yf

def download_stock(stock):
	""" try to query the iex for a stock, if failed note with print """
	print(stock)
	stock_df = yf.download(tickers = stock.upper(), period = 'max', interval = '1d')
	stock_df['Name'] = stock
	output_name = 'data/nasdaq100/' + stock + '.csv'
	print(len(stock_df))
	stock_df.to_csv(output_name)

if __name__ == '__main__':

	""" set the download window """
	now_time = datetime.now()
	nasdaq100 = list(pd.read_csv("data/nasdaq100.csv")["Symbol"])
	print(len(nasdaq100))

	"""here we use the concurrent.futures module's ThreadPoolExecutor
		to speed up the downloads buy doing them in parallel 
		as opposed to sequentially """
	for stock in nasdaq100:
		download_stock(stock)

	#timing:
	finish_time = datetime.now()
	duration = finish_time - now_time
	minutes, seconds = divmod(duration.seconds, 60)
	print(f'The script took {minutes} minutes and {seconds} seconds to run.')
