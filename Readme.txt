

web_scrapping_time.py: Access news articles from Estadão, Folha and Valor to extract information. It's necessary to login in each newspaper.

aws_blob.py: Download data from AWS to local file.

epu_bbk.py: Computes the Economic Policy Uncertainty (EPU) Index for Brazil based on Baker, Bloom and Davis (2016). 

har.py: This is the main file of the work. Load, preprocess, implement the loop reestimating the model at every iteration (every day) and present the main results. We consider eight different models (har, har-cj, lhar-cj, lhar-cj+, ar1, harq, HARX, HARX (AdaLasso)) and six different assets (BOVA11, PETR4, VALE3, BBAS3, ITUB4, BBDC4) for four forecasting horizons (1, 5 ,10, 22 days).

estimators.py: Compute estimators robust to microstrucutre noise based on Hautsch and Podolski (2013). The quadratic variation is partitioned into a continuous component (Integrated Variance) and a Jump Component. Include implementation of Lee and Mykland (2012) jump test taking into account multiple testing issues. Considering a database of tick-by-tick data for all transaction each day, computations are based on a sampling frequency of 1-minute, following Hautsch and Podolski (2013).

news_count.py: Computes firm specific news based on Fernandes and Pereira (2022).

