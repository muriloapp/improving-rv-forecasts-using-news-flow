# improving-rv-forecasts-using-news-flow

web_scrapping.py: access news articles from Estadão, Folha and Valor to extract information.
data.py: download data from Azure.
epu.py: computes the Economic Policy Uncertainty (EPU) Index for Brazil based on Baker, Bloom and Davis (2016). 
har.py: load data, preprocess and implement the main model.
estimators.py: compute rv estimators robust to microstrucutre noise based on Hautsch and Podolski (2013). Include implementation of Lee and Mykland (2012). Considering a database of tick-by-tick data for all transaction each day, computations are based on a sampling frequency of 1-minute, following Hautsch and Podolski (2013).
news_count.py: computes firm specific news based on Fernandes and Pereira (2022).
