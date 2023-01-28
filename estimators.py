#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:34:35 2021

@author: muriloandreperespereira

Compute estimators robust to microstrucutre noise based on Hautsch and Podolski (2013). The quadratic variation is 
partitioned into a continuous component (Integrated Variance) and a Jump Component.
Include implementation of Lee and Mykland (2012) jump test taking into account multiple testing issues. 

Considering a database of tick-by-tick data for all transaction each day, computations are based on a sampling frequency of 1-minute, following Hautsch and Podolski (2013).

"""

import pandas as pd
import numpy as np
import datetime 
from datetime import datetime
from datetime import timedelta
from datetime import time
from azure.storage.blob import BlobServiceClient
from zipfile import ZipFile
import io
import os
from numba import prange
import numba




@numba.njit(cache=False, parallel=False, fastmath=False)
def _preaverage(data, weight):
    """
    
    Preaverage an observation matrix with shape = (n, p) given a weight vector
    with shape = (K-1, p).

    Parameters
    ----------
    data : numpy.ndarray, shape = (n, p)
        The observation matrix of synchronized log-returns.
    weight : numpy.ndarray, shape = (K-1, )
        The weight vector, looking back K -2 time steps.

    Returns
    -------
    data_pa : numpy.ndarray, shape = (n, p)
        The preaveraged returns.


    """

    n, p = data.shape
    K = weight.shape[0] + int(1)
    data_pa = np.full_like(data, np.nan)
    for i in prange(K-1, n):
        for j in range(p):
            data_pa[i, j] = np.dot(weight, data[i-K+2:i+1, j])
    return data_pa


@numba.njit
def _numba_minimum(x):
    """
    
    The weighting function of Christensen et al. (2010).
    
    """
    return np.minimum(x, 1-x)


def mrc_fun(tick_series_list, theta=None, g=None, bias_correction=True, k=None):
    
    if g is None:
        g = _numba_minimum

    #p = len(tick_series_list)

    data = tick_series_list[0]
    data = np.diff(data.to_numpy(), axis=0)[:, None]

    cov = _mrc(data, theta, g, bias_correction, k)

    return cov


#@numba.njit(cache=False, fastmath=False, parallel=False)
def _mrc(data, theta, g, bias_correction, k):
    """
    
    Compute the Integrated Variance, Jump Contribution and Integrated Quarticity.
    
    """
    
    
    n, p = data.shape

    # Get the bandwidth
    if k is not None and theta is not None:
        raise ValueError("Either ``theta`` or ``k`` can be specified,"
                         " but not both! One of them must be ``None``.")
    if k is None:
        if theta is None:
            theta = 0.4
        k = _get_k(n, theta)

    if theta is None:
        if bias_correction:
            theta = k / np.sqrt(n)
        else:
            theta = k / np.power(n, 0.6)

    # If theta is greater than zero comute the preaveraging estimator, otherwise the estimator is just the realized covariance matrix.
    
    psi2 = np.sum(g(np.arange(1, k)/k)**2)/k
    psi1 = np.sum((g(np.arange(1, k)/k)-g((np.arange(1, k)-1)/k))**2)*k

    weight = g(np.arange(1, k)/k)
    data_pa = _preaverage(data, weight)

    data_pa = data_pa.flatten()
    data_pa = data_pa[~np.isnan(data_pa)]
    data_pa = data_pa.reshape(-1, p)

    # The bias correction term, bc, needs to be initialized as array to have a consistent type for numba.
    bc = np.zeros((p,p))

    if bias_correction:
        bc += psi1 / (theta ** 2 * psi2) * data.T @ data / n / 2

    finite_sample_correction = n / (n - k + 2)
    
    mrc = finite_sample_correction / (psi2 * k) * data_pa.T @ data_pa - bc
        
   
    # Compute BT
    mu = np.sqrt(2/np.pi) # Expectativa normal padrão
    bt = finite_sample_correction / (psi2 * k * mu**2) * np.absolute(data_pa[:-k].T) @ np.absolute(data_pa[k:]) - bc
    btv = finite_sample_correction / (k * psi2) * (data_pa.T @ data_pa - 1/mu**2 * np.absolute(data_pa[:-k].T) @ np.absolute(data_pa[k:]))
    rv = data.T @ data 
    
    
    # Estimator of Integrated Quarticity (It will be important latter)
    for i in range(len(data)-2*k):
        j = i + k + 1
        aux =sum(data[j:i+2*k]**2)
    
    # Realized Quarticity
    rq = 1/(3 * theta**2 * psi2**2) * sum(data_pa**4)  -  1/n * 2 * psi1/(2 * theta**4 * psi2**2) * sum(data_pa[:-k]**2 * aux ) +  1/(n*4) * (psi1**2/(theta**4 * psi2**2)) * sum((data[:-2]**2) * data[2:]**2)
    


    return rv, mrc, bt, btv, rq


@numba.njit
def _get_k(n, theta, bias_correction=True):
    """ 
    
    Get the optimal bandwidth for preaveraging depending on the sample
    size and whether or not to correct for the bias.
    
    """
    if theta > 0:
        if bias_correction:
            k = np.ceil(np.sqrt(n)*theta)
        else:
            delta = 0.1
            k = np.ceil(np.power(n, 0.5+delta)*theta)
    else:
        k = 1

    return int(k)




def previous(items, items_price, pivot):
                  xxx = min(items[:], key=lambda x: abs(x - pivot))
                 
                  idx = np.where(items[:]==xxx)
                  d = float(items_price[idx])
                  return d,xxx




def compute_intervals(items,items_price,pivot):
                   
            # Find the nearest price
            previous_price = np.empty((len(pivot),1))
            avg_time_delay = []
            for j in range(len(pivot)):
                try:
                    idx = np.where(pivot[j]<=items)[0][0]+1
                    if idx <=0: idx =1
                except: idx = len(items)
                d,xxx = previous(items[0:idx],items_price[0:idx],pivot[j])
                previous_price[j]=d
                avg_time_delay.append((pivot[j]-xxx)/1000000000)
            time_delay = np.mean(abs(np.array(avg_time_delay[5:-60],dtype=float)))
            
            
            return previous_price,time_delay


def LeeMykland(intervals,bt):
        """
        
        Jump test from Lee and Mykland (2012).
    
        """
        
    
        P_tilde = np.log(intervals['nearest_price'])
        
        k=1 # Maximum lag fora da banda, fazer acf
        n=len(intervals)
        
        # Get n-k
        n_diff = n - k
        
        # - means lead
        P_tilde_shift = P_tilde.shift(-k)
        
        # Calculate q_hat
        q_hat =  np.sqrt(1/(2 * n_diff) * sum((P_tilde[1:n_diff] - P_tilde_shift[1:n_diff])**2)) 
        
        # Choose optimal C
        C = 1/8
        if q_hat*100<=0.8: C=1/9
        if q_hat*100<=0.09: C=1/18
        if q_hat*100<=0.05: C=1/19
        
        
        if np.floor(C*np.sqrt(n/k)) == 0: 
                  M = 1
        else:  
            M = np.floor(C*np.sqrt(n/k))
            
            
        P_tilde_t_ik =pd.Series([0]*int(len(P_tilde)/k),dtype=float)
        for i in range(int(len(P_tilde)/k)):
            P_tilde_t_ik[i] = float(P_tilde[i*k])
            
            
        P_hat_tj=pd.Series([0]*int(len(P_tilde)/k),dtype=float)
        for i in range(len(P_hat_tj)) : 
          P_hat_tj[i]= np.mean(P_tilde_t_ik[int(np.floor(i/k)):int(np.floor(i/k)+M)])  
          
        
        drop = []
        for i in range(len(P_hat_tj)-1):
            if P_hat_tj.index[i] % M==1:
                drop.append(P_hat_tj.index[i])
                
        P_hat_tj=P_hat_tj.drop( drop,axis=0)
        
        P_hat_tj = P_hat_tj.reset_index()
        P_hat_tj = P_hat_tj.drop(['index'],axis=1)
        
        # Compute statistic
        L_tj = P_hat_tj[1:].values - P_hat_tj[0:-1].values
        
        
        # Calculate limit
        # V_n = np.var(np.sqrt(M)*L_tj)
        
        sigma_hat = np.sqrt(bt) # robust estimator --> sugested in Lee and Mykland 2012
        T=1 # No need of adjustment to T, based bt
        plim_Vn = 2/3 * sigma_hat**2 * C**2 * T + 2 * q_hat**2
        # Calculte Chi_tj
        Chi_tj = np.sqrt(M) / np.sqrt(plim_Vn) * L_tj
            
        # Define A_n & B_n
        A_n = np.sqrt(2 * np.log(np.floor(n/(k*M)))) - (np.log(np.pi) + np.log ( np.log(np.floor(n/(k*M))) )) /( 2 * np.sqrt(2 * np.log(np.floor(n/(k*M)))))
        B_n = 1 / (np.sqrt(2 * np.log(np.floor(n/(k*M)))))
              
        # Define Xi_hat
        #Xi_hat = B_n**(-1) * (abs(Chi_tj) - A_n)
        Xi_hat_max =  B_n**(-1) * (max(abs(Chi_tj)) - A_n)
          
        # Jump threshold
        significance_level = 0.01
        beta_star   =  -np.log(-np.log(1-significance_level)) # Jump threshold
  
        # Correcting for multiple tests based on Baj
        
        confint = np.sqrt(2*np.log(n))
        
        
        if Xi_hat_max>beta_star :
            if max(Chi_tj) < -confint or max(Chi_tj) > confint:
                J=1
            else:
                J=0       
        else: J=0
        
        return J, max(L_tj)
    
    
    
    
# In[1]:
    
#####################
#######  Run ########
#####################


stock_list = ['BOVA11', 'PETR4', 'VALE3', 'BBAS3', 'ITUB4', 'BBDC4']

# For each stock we compute the estimators based on 1-minute data. It takes approximately 40 hours to run for all six stocks.

# Once the data is in a local file, it's not necessary to consider this.
# STORAGEACCOUNTURL = "https://murilopereira.blob.core.windows.net"
# STORAGEACCOUNTKEY = "xxxxxxxxxxxxxxx"
# container_name = "data"
# blob_service_client= BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
    

# Empty list to store results
realized_vol = []
previous_time_delay = []

path = 'xxxxxxxx'
dates = pd.read_excel(path + '/date.xlsx') # Trading dates
stock = 'VALE3' # Run for each stock on stock_list
os.chdir(path + '/Data/' + stock)
for i in range(len(dates)):
        
        blobname = 'NEG/' + str(dates.iloc[i,0]) + '/' + stock + '.zip'
    
        # Download blob to local file. Done!
        
        # local_file_path = blobname
        # with open(str(dates.iloc[i,0]) + '.zip', "wb") as download_file:
        #     blob_client = blob_service_client.get_blob_client(container=container_name, blob=blobname)
        #     download_file.write(blob_client.download_blob().readall())
    
        
    
    
        # Open zipped files for each day and preprocess
        with ZipFile(path + '/Data/' + stock + '/' + str(dates.iloc[i,0]) + '.zip') as myzip:
            with myzip.open(stock+'.txt') as myfile:
                aux = myfile.read()
    
    
    
        buf_bytes = io.BytesIO(aux)
        headers = ['date', 'col2', 'col3', 'price1','price2', 'col6', 'time', 'col7','col8', 'col9', 'col10', 'col11','col12', 'col13', 'col14', 'col15','col16', 'col17', 'col18','col19','col20', 'col21', 'col22']
        dtypes = {'date':str, 'col2':str, 'col3':str, 'price1':str,'price2':str, 'col6':str, 'time':'str', 'col7':'str','col8':'str', 'col9':'str', 'col10':'str', 'col11':'str','col12':'str', 'col13':'str', 'col14':'str', 'col15':'str','col16':'str', 'col17':'str', 'col18':'str','col19':'str','col20':'str', 'col21':'str', 'col22':'str'}
        parse_dates = [['date','time']]
        df = pd.read_csv(buf_bytes, sep='[;|.]', header=None,  names=headers, dtype=dtypes, parse_dates=parse_dates, engine='python')
        
        
        # Start work with df
        df = pd.concat([df.iloc[:,0],df.iloc[:,3],df.iloc[:,4]],axis=1)
        df = df.drop_duplicates(subset="date_time") # Drop duplicates
        df['price'] = 0 
        for j in range(len(df)):
                df.iloc[j,3] = float(str(df.iloc[j,1])+ '.'+str(df.iloc[j,2]))
        df =df.drop(['price1','price2'],axis=1)    
        
        # Reindex
        df = df.reset_index()
        df = df.drop(['index'],axis=1)
        
        # Easy way to exclude after market period and also considering different market working time during summer period
        drop=[]
        for j in range(len(df)):
            if  df.iloc[j,0].hour==18:
                drop.append(j)
        if drop == []:
            for j in range(len(df)):
                if  df.iloc[j,0].hour==17:
                    drop.append(j)
                    
        df = df.drop(drop,axis=0)
        
        date = str(dates.iloc[i,0])
        date = date[:4]+'-'+date[4:6]+'-'+date[6:]
        
        
        # Sampling Frquency
        aux_time = datetime.strptime(date +' 10:05:00.000000', '%Y-%m-%d %H:%M:%S.%f')
        intervals = [aux_time]
        for j in range(1,471): # Automatically consider when market close at 5pm (410)
                temp1 = aux_time+timedelta(minutes=j)#5
                intervals.append(temp1)
        intervals = pd.DataFrame(intervals)     
        
        
        # Increase speed
        pivot = np.array(intervals.iloc[:,0],np.datetime64)
        items =  np.array(df.iloc[:,0],np.datetime64)
        items_price = np.array(df.iloc[:,1])
        previous_price,time_delay = compute_intervals(items,items_price,pivot)
        previous_time_delay.append(time_delay)
        
        
        intervals['nearest_price']=previous_price
    
        print(date)
        
        tick_series_list = [pd.Series(np.log(intervals['nearest_price']))]
        theta = 0.6
        
        # Computing values
        rv, mrc, bt, btv,rq = mrc_fun(tick_series_list,theta = theta,g=_numba_minimum,bias_correction=True)
        J,L_tj = LeeMykland(intervals, bt)

        realized_vol.append((date,float(rv),float(mrc),float(bt),float(btv),float(rq),J,float(L_tj)))


# Save
# with open('VALE3_1min_rq.data','wb') as f:
#                  pickle.dump(realized_vol,f)
        





