#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 16:59:20 2021

@author: muriloandreperespereira


This is the main file of the work. Load, preprocess, implement the main model.

We consider eight different models (har, har-cj, lhar-cj, lhar-cj+, ar1, harq, HARX, HARX (AdaLasso)) and six different assets (BOVA11, PETR4, VALE3, BBAS3, ITUB4, BBDC4) for 
four forecasting horizons (1, 5 ,10, 22 days).

"""


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import datetime
from datetime import datetime
import pickle
import scipy
from datetime import timedelta 
#import nltk
#from nltk.tokenize import word_tokenize
#import pt_core_news_sm
#from nltk.stem import RSLPStemmer
#import asgl
from arch.bootstrap import MCS
import os
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from arch.unitroot import ADF
import warnings
warnings.filterwarnings('ignore')


class main:
    
    def __init__(self):
        self.end_train_str = '2018-03-01' # Training period
        self.expanding_window = True
        self.eval = 'MAE' 


    def build_model(self,_alpha, _l1_ratio):
        estimator = ElasticNet(
            alpha=_alpha,
            l1_ratio=_l1_ratio,
            fit_intercept=True,
            normalize=True,
            max_iter=500,
            copy_X=True,
            warm_start=False,
            positive=False,
            random_state=None,
        )
        
        return estimator
        
    
    def build_ridge(self,_alpha,):
        estimator = ElasticNet(
            alpha=_alpha,
            fit_intercept=True,
            normalize=True,
            max_iter=500,
            copy_X=True,
            warm_start=False,
            positive=False,
            random_state=None,
        )

        return estimator
    
    
    def load_df(self, stock):
        """
        Load dataframe with Realized Volatility, Modulated Realized Volatility, Integrated Variance, Jump Component, Realized Quarticity, J (Dummy for jumps) and
        jump (results based of Lee and Mykland (2012)).

        Parameters
        ----------
        stock : stock name

        Returns
        -------
        df : Dataframe containing rv, mrv, bt, btv, rq, J and jump 

        """
        
        # Change diretory
        path = 'xxxxx'
        os.chdir(path)
        
        # Loading preprocessed data from pickle files
        with open(stock+'_1min_rq.data','rb') as f:
                 data = pickle.load(f)

        # Dataframe
        df = pd.DataFrame(data,columns=['date','rv','mrv','bt','btv','rq','J','jump'])
        
        # Datetime
        for i in range(0, len(df)):
                df['date'].iloc[i] = datetime.strptime(str(df['date'].iloc[i]), '%Y-%m-%d').date()
        df = df.set_index('date')
        
        # Zero bound
        df = df.iloc[:,0:5]
        for i in range(len(df)):
                    if df.iloc[i,3]<=0:
                        df.iloc[i,3] = df['btv'][df['btv']>0].min()
        
        df = df.fillna(method='ffill')
        #df = df.fillna(0)
        
        return df
    

    def compute_ma(self, df, bt=True, btv=False, ret = False):
         """
         Compute Moving Averages.
    
         Parameters
         ----------
         df : Pandas dataframe. Dataframe containing rv, mrv, bt, btv, rq, J and jump.
         bt : Boolean, optional
             True to compute MA for the continuous component. The default is True.
         btv : Boolean, optional
             True to compute MA for the jump component. The default is False.
         ret : Boolean, optional
             True to compute MA for the returns component. The default is False.
    
         Returns
         -------
         Dataframe data containing moving averages and/or returns.
    
         """
         
         data = pd.DataFrame()
         if bt == True:
             
             # Continuous component
             data = pd.DataFrame(df['bt'])
             data = data.rename({'bt':'bt_1'},axis='columns')
             # Compute bt lags
             data['bt_5'] = data['bt_1'].rolling(5).mean()
             data['bt_22'] = data['bt_1'].rolling(22).mean()   
         
         if btv == True:
             
             # Jump component
             data['j_1'] = df['btv']
             data['j_5'] = data['j_1'].rolling(5).mean()
             data['j_22'] = data['j_1'].rolling(22).mean()
             
         if ret == True:
             
             # Adding leverage effects
             path = 'xxxxxxxx' # path returns
             returns = pd.read_excel(path + stock[j]+'_1d.xlsx',index_col='date')
             #returns = pd.read_excel('/Users/muriloandreperespereira/Desktop/Thesis/Vol forecast/rv/Data/returns/'+stock[j]+'_1d.xlsx',index_col='date')
             returns['returns_5'] = returns['returns'].rolling(5).mean() ## Change name
             returns['returns_22'] = returns['returns'].rolling(22).mean()
             
             returns[returns>0]=0     
             returns = returns.iloc[21:,:]   
             returns = abs(returns)
             
             
         # Remove initial period
         data = data.iloc[21:,:]
         
         
         if ret == False:
             return data
         else:
             return data, returns
     



    def har(self, df, f_horizon, model='har', exog=None, regularize=None):
        """
        This is the main method. It preprocess the data, implements the loop reestimating the model at every iteration (every day) 
        and presents the main results.

        Parameters
        ----------
        df : Pandas dataframe
            Pandas dataframe. Dataframe containing rv, mrv, bt, btv, rq, J and jump.
        f_horizon : Integer
            Forecasting horizon.
        model : String, optional
            The default is 'har'.
        exog : Pandas dataframe, optional
            The default is None.
        regularize : String, optional
            Three types of regularization: Adaptive Lasso, Lasso and Bayesian Ridge. The default is None.

        Returns
        -------
            Returns mae, mafe, fe_ts, betas, y_hat, std in case of regularization and mae, mafe, fe_ts, y_hat, std otherwise.

        """
    
        # Preaveraged estimator. We will to make forecasts for this quantity
        mrv = pd.DataFrame(df['mrv'])
      
        # Compute lags 
        mrv['mrv_1'] = mrv['mrv'] # In our setting mrv_1 = mrv since we will make prediction for h+
        mrv['mrv_5'] = df['mrv'].rolling(5).mean()
        mrv['mrv_21'] = df['mrv'].rolling(22).mean()
        mrv = mrv.iloc[21:,:]
     
        # Adjust the database for each model
        if model == 'har-cj':            
            data = self.compute_ma(df, bt=True, btv=True, ret = False)
        
        if model == 'char':            
            data = self.compute_ma(df, bt=True, btv=False, ret = False)
        
        if model == 'lhar-cj':            
            data, returns = self.compute_ma(df, bt=True, btv=True, ret = True)                   
        
        if model == 'lhar-cj+':            
            data, returns = self.compute_ma(df, bt=True, btv=True, ret = True)
                
            # Signed jumps as in Sheppard and Patton (2011)
            jc = pd.DataFrame(data.iloc[:,3])
            jc['jc+'] = pd.DataFrame(data.iloc[:,3])
            for i in range(len(jc)):
                if returns.iloc[i,0]==0:
                    jc.iloc[i,1] = 0
                else:
                    jc.iloc[i,0] = 0
            data = data.drop(['j_1'],axis=1)
            data = pd.concat([data,jc],axis=1)
                        
        if model == 'lhar':            
            data, returns = self.compute_ma(df, bt=False, btv=False, ret = True)
              
        if model =='harq':    
            #  rq*mrv as in Bollerslev et al (2016)
            data = mrv
            data = data.drop(['mrv'],axis=1)
            data.insert(3, 'rq*mrv', df['rq']*data['mrv_1'])
            
        
        
        
        
        # Set training period
        end_train = np.where(mrv.index >= datetime.strptime(self.end_train_str,'%Y-%m-%d').date())[0].min() 
        # Store results
        y_hat_list = []
        betas      = []
        
        def multiple(m, n):
            return True if m % n == 0 else False
        
        
        m=1
        # Start the loop, models are estimated every day. We forecast the mean over the next h days
        for l in range(len(mrv)-end_train):    
            
                ## Building y based entirely on mrv dataframe 
                
                if f_horizon == 1:
                    y_train = np.log(np.asarray(mrv.iloc[f_horizon:end_train+l,0],dtype=float)*m)   
                    y_test = pd.DataFrame(mrv.iloc[end_train+l:,0])
               
                if f_horizon == 5:
                    y_train = np.log(np.asarray(mrv.iloc[f_horizon:end_train+l,2],dtype=float)*m)   
                    y_test = pd.DataFrame(mrv.iloc[end_train+l:,2])
                    
                if f_horizon == 10:
                    mrv['mrv_10'] = df['mrv'].rolling(10).mean()
                    y_train = np.log(np.asarray(mrv.iloc[f_horizon:end_train+l,4],dtype=float)*m)   
                    y_test = pd.DataFrame(mrv.iloc[end_train+l:,4])   
                    
                if f_horizon == 22:
                    y_train = np.log(np.asarray(mrv.iloc[f_horizon:end_train+l,3],dtype=float)*m)   
                    y_test = pd.DataFrame(mrv.iloc[end_train+l:,3])
                    
           
            
           
                ## Buiding and transforming X                 
                
                if model == 'har': # Base model Corsi (2009)
                    X_train =  np.log(pd.concat([mrv.iloc[:end_train+l-f_horizon,1:4]],axis=1).iloc[:,0:3]*m) 
                    X_test = np.log(pd.concat([mrv.iloc[(end_train+l-f_horizon):-f_horizon,1:4]],axis=1).iloc[:,0:3]*m)  
          
                if model == 'har-cj': # Including Jumps
                    X_train = pd.concat([data.iloc[:end_train+l-f_horizon,0:len(data.columns)]],axis=1)
                    X_test = pd.concat([data.iloc[end_train+l-f_horizon:-f_horizon,0:len(data.columns)]],axis=1)
                    X_train.loc[:,['bt_1','bt_5','bt_22']] = np.log(X_train.loc[:,['bt_1','bt_5','bt_22']]*m)  
                    X_test.loc[:,['bt_1','bt_5','bt_22']] = np.log(X_test.loc[:,['bt_1','bt_5','bt_22']]*m)  
                    X_train.loc[:,['j_1','j_5','j_22']] = np.log(1+X_train.loc[:,['j_1','j_5','j_22']]*m) 
                    X_test.loc[:,['j_1','j_5','j_22']] = np.log(1+X_test.loc[:,['j_1','j_5','j_22']]*m)    
        
                if model == 'char': # Continuous HAR
                    X_train = pd.concat([data.iloc[l:end_train+l-f_horizon,0:len(data.columns)]],axis=1)
                    X_test = pd.concat([data.iloc[end_train+l-f_horizon:-f_horizon,0:len(data.columns)]],axis=1)
            
                if model == 'lhar-cj': # Leveraged HAR with jumps
                    X_train = pd.concat([data.iloc[:end_train+l-f_horizon,0:len(data.columns)],returns.iloc[:end_train+l-f_horizon,0:len(returns.columns)]],axis=1)
                    X_test = pd.concat([data.iloc[end_train+l-f_horizon:-f_horizon,0:len(data.columns)],returns.iloc[end_train+l-f_horizon:-f_horizon,0:len(returns.columns)]],axis=1)
                    X_train.loc[:,['bt_1','bt_5','bt_22']] = np.log(X_train.loc[:,['bt_1','bt_5','bt_22']]*m)  
                    X_test.loc[:,['bt_1','bt_5','bt_22']] = np.log(X_test.loc[:,['bt_1','bt_5','bt_22']]*m)  
                    X_train.loc[:,['j_1','j_5','j_22']] = np.log(1+X_train.loc[:,['j_1','j_5','j_22']]*m) 
                    X_test.loc[:,['j_1','j_5','j_22']] = np.log(1+X_test.loc[:,['j_1','j_5','j_22']]*m)    
                    
                if model == 'lhar-cj+': # Leveraged HAR with signed jumps
                    X_train = pd.concat([data.iloc[:end_train+l-f_horizon,0:len(data.columns)],returns.iloc[:end_train+l-f_horizon,0:len(returns.columns)]],axis=1)
                    X_test = pd.concat([data.iloc[end_train+l-f_horizon:-f_horizon,0:len(data.columns)],returns.iloc[end_train+l-f_horizon:-f_horizon,0:len(returns.columns)]],axis=1)
                    X_train.loc[:,['bt_1','bt_5','bt_22']] = np.log(X_train.loc[:,['bt_1','bt_5','bt_22']]*m)  
                    X_test.loc[:,['bt_1','bt_5','bt_22']] = np.log(X_test.loc[:,['bt_1','bt_5','bt_22']]*m)  
                    X_train.loc[:,['j_1','j_5','j_22']] = np.log(1+X_train.loc[:,['j_1','j_5','j_22']]*m) 
                    X_test.loc[:,['j_1','j_5','j_22']] = np.log(1+X_test.loc[:,['j_1','j_5','j_22']]*m)             
                    
                if model == 'lhar': # Leveraged HAR 
                    X_train = pd.concat([mrv.iloc[:end_train+l-f_horizon,1:4],returns.iloc[:end_train+l-f_horizon,0:len(returns.columns)]],axis=1)
                    X_test = pd.concat([mrv.iloc[end_train+l-f_horizon:-f_horizon,1:4],returns.iloc[end_train+l-f_horizon:-f_horizon,0:len(returns.columns)]],axis=1)
                    X_train.loc[:,['returns','returns_5','returns_22']] = np.log(X_train.loc[:,['returns','returns_5','returns_22']]*m)  
                    X_test.loc[:,['returns','returns_5','returns_22']] = np.log(X_test.loc[:,['returns','returns_5','returns_22']]*m)  
                    
                if model == 'harq': # HAR with realized quarticity (Bollerslev et al (2016))
                    X_train = pd.concat([data.iloc[:end_train+l-f_horizon,:]],axis=1)
                    X_test = pd.concat([data.iloc[(end_train+l-f_horizon):-f_horizon,:]],axis=1)
                    X_train = np.log(X_train*m)  
                    X_test = np.log(X_test*m)  
            
                if model == 'ar1': # Random walk
                    X_train = np.asarray(pd.concat([mrv.iloc[:end_train+l-f_horizon,1]],axis=1))
                    X_test = np.asarray(pd.concat([mrv.iloc[end_train+l-f_horizon:-f_horizon,1]],axis=1))
                    
                    # Final adjustments
                    X_train=np.array(X_train)
                    X_train[:,0] = np.log(X_train[:,0]*m)      
                    X_test=np.array(X_test)
                    X_test[:,0] = np.log(X_test[:,0]*m)      
                    
                    
                
                # Add exogenous regressors (text-based variables)
                if exog is not None: 
                     
                     X_train = pd.concat([X_train,exog.iloc[:end_train+l-f_horizon,0:len(exog.columns)]],axis=1)
                     X_test = pd.concat([X_test,exog.iloc[end_train+l-f_horizon:-f_horizon,0:len(exog.columns)]],axis=1)
            
            
                if regularize is None: # HARX in Fernandes and Pereira (2022)                 
            
                    X_train = sm.add_constant(X_train)
                    X_test  = sm.add_constant(X_test, has_constant='add')
                    res_train = sm.OLS(y_train,X_train).fit()
                    
                    #print(res_train.summary())   
                    #new = res_train.get_robustcov_results(cov_type='HAC',maxlags=5)
                    #new.summary()
                    #scipy.stats.jarque_bera(y_train)
                                  
                    y_hat = np.exp(res_train.predict(X_test)) #*np.exp(np.var(res_train.resid)/2)
                    y_hat_list.append(y_hat[0]) # Append the prediction of one day ahead
                    
                    if l==0:
                        mae = np.mean(abs(res_train.resid))
                    
                
                if regularize == 'Adalasso': # HARX (AdaLasso) in Fernandes and Pereira (2022)        
                # First step: defining weights based on ridge regression, as sugested by Zou (2006) in case of possible multicolinearity. 
                # Next consider the LARS algorithm
                         
                         if l == 0 :    
                                
                                ridge = self.build_ridge(_alpha=1.0)
                                kfcv = TimeSeriesSplit(n_splits=5)
                                alpha_list = [0.001, 0.01, 0.1, 0.5]
                               #l1_ratio_list = [0,0]#list(np.arange(0.1,0.6,0.1))
                                params = [{'alpha':alpha_list}]
                                    
                                finder = GridSearchCV(
                                estimator=ridge,
                                param_grid=params,
                                refit=False,
                                cv=kfcv,  # change this to the splitter subject to test
                                verbose=1,
                                scoring='r2',#'neg_mean_absolute_error',
                                pre_dispatch=8,
                                error_score=-999,
                                return_train_score=True
                                )
                            
                                finder.fit(X_train, y_train)
                                best_params = finder.best_params_
                                
                         alpha_ridge = best_params['alpha']
                         gamma = 1
                         ridge = self.build_ridge(_alpha=alpha_ridge)
                         ridge.fit(X_train, y_train)  
                         weight = np.power(np.abs(ridge.coef_), gamma)
                         
                # If one wants to fine-tune the model at every x number of iterations
                         # gamma = 1
                         # #if l==0 or multiple(l, 20):   
                         # # Defining weights based on OLS                            
                         # reg = LinearRegression(normalize=True).fit(X_train,y_train)
                         # weight = np.power(np.abs(reg.coef_), gamma) 
                             
                         
                # Adaptive Lasso
                         if l==0 or multiple(l, 1):
                               
                               Elast = self.build_model(_alpha=1.0, _l1_ratio=0.3)
                               kfcv = TimeSeriesSplit(n_splits=5)
                               #scores = cross_val_score(Elast, X_train, y_train, cv=kfcv, scoring='neg_mean_absolute_error')
                               alpha_list = [0.001,0.01,0.1]
                               l1_ratio_list = [0.1,0.5,1]
                               params = [{'alpha':alpha_list,'l1_ratio':l1_ratio_list}]
                                    
                               finder = GridSearchCV(
                                   estimator=Elast,
                                   param_grid=params,
                                   refit=False,
                                   cv=kfcv,  # change this to the splitter subject to test
                                   verbose=1,
                                   scoring='r2',#'neg_mean_absolute_error',
                                   pre_dispatch=8,
                                   error_score=-999,
                                   return_train_score=True
                                   )
                            
                               finder.fit(X_train * weight, y_train)                   
                               best_params = finder.best_params_
                               alpha = best_params['alpha']
                            
                         Elast = self.build_model(_alpha=alpha, _l1_ratio=1)
                         Elast.fit(X_train * weight, y_train)
                         coef = Elast.coef_ *weight
                         coef = np.expand_dims(coef, axis=1)                        
                         coef = np.insert(coef,0,Elast.intercept_)
                         betas.append(coef)
                         
                         X_test = sm.add_constant(X_test, has_constant='add')
                         y_hat = np.exp( X_test @ coef ) #*np.exp(np.var(resids)/2)
                         y_hat_list.append(y_hat[0])
                         
                         if l ==0:
                             X_train = sm.add_constant(X_train, has_constant='add')
                             mae = np.mean(abs(X_train @ coef-y_train))
                             
                
                if regularize == 'Lasso':
                    
                    if l==0 or multiple(l, 20):  
                            
                            Elast = self.build_model(_alpha=1.0, _l1_ratio=1)
                            kfcv = TimeSeriesSplit(n_splits=5)
                            alpha_list = [0.00001,0.0001,0.001,0.01,0.1]
                            l1_ratio_list = [1,1]
                            params = [{'alpha':alpha_list,'l1_ratio':l1_ratio_list}]        
                            finder = GridSearchCV(
                            estimator=Elast,
                            param_grid=params,
                            refit=False,
                            cv=kfcv,  # change this to the splitter subject to test
                            verbose=1,
                            scoring='neg_mean_absolute_error',
                            pre_dispatch=8,
                            error_score=-999,
                            return_train_score=True
                            )
                    
                            finder.fit(X_train, y_train)        
                            best_params = finder.best_params_                    
                            alpha = best_params['alpha']
                            l1_ratio = best_params['l1_ratio']
                    
                    Elast = self.build_model(_alpha=alpha, _l1_ratio=l1_ratio)
                    Elast.fit(X_train, y_train)            
                    betas.append(Elast.coef_)            
                    y_hat = np.exp(Elast.predict(X_test))#*np.exp(np.var(res_train.resid)/2)
                    y_hat_list.append(y_hat[0])
                    
                
                
                if regularize == 'BayesianRidge':
                    
                    clf = BayesianRidge(compute_score=True,normalize=True)
                    clf.fit(X_train, y_train)
                    #betas.append((Elast.intercept_,Elast.coef_))
                    y_hat = np.exp(clf.predict(X_test))#*np.exp(np.var(res_train.resid)/2)
                    y_hat_list.append(y_hat[0])  
                             
                             
    
        y_hat = pd.DataFrame(y_hat_list,index=mrv.index[end_train:])
        if f_horizon==1: y_test = pd.DataFrame(mrv.iloc[end_train:,0])
        elif f_horizon==5: y_test = pd.DataFrame(mrv.iloc[end_train:,2])
        elif f_horizon==10: y_test = pd.DataFrame(mrv.iloc[end_train:,4]) 
        elif f_horizon==22: y_test = pd.DataFrame(mrv.iloc[end_train:,3])
        
        
        final = pd.concat([y_test,y_hat],axis=1) # Concat both series 
        if self.eval == 'MAE':
            final['FE'] = (final.iloc[:,1]-final.iloc[:,0]*10000)
        elif self.eval == 'MAPE':
            final['FE'] = (final.iloc[:,1]-final.iloc[:,0]*10000)/(final.iloc[:,0]*10000)
    
        # Final calculations
        fe_ts = abs(final['FE'])
        mafe = np.mean(abs(final['FE']))
        std = np.std(abs(final['FE']))


        if regularize is  None:
            return mae, mafe, fe_ts, y_hat, std
        else:
            return mae, mafe, fe_ts, betas, y_hat, std
    




# In[1]:

    
##########################################
###### Start the engine and run  #########
##########################################

main = main()

stock     = ['BOVA11', 'PETR4', 'VALE3', 'BBAS3', 'ITUB4', 'BBDC4']
f_horizon = [1,5,10,22]
model     = ['har', 'har-cj', 'lhar-cj', 'lhar-cj+', 'ar1', 'harq', 'HARX', 'HARX (AdaLasso)']

# Empty lists
mae_results   = []
mafe_results  = []
std_mafe      = []
y_hat_results = []
mcsa_results  = []
betas_results = []
db_losses     = []

os.chdir('xxxxxxxx') # Diretory containing the data

for f in f_horizon:  
    for j in range(len(stock)):
        
        df = main.load_df(stock[j]) # Load dataframes
        losses = []                 # Store losses for the Model Confidence Set
        
        for i in range(len(model)):
            
            if model[i] != 'HARX' and model[i] != 'HARX (AdaLasso)':
                mae, mafe, fe_ts, y_hat, std = main.har(df= df, f_horizon = f, model = model[i]) 
            
            else:
                # Load text-based quantities computed before for extended models
                with open('count_'+stock[j]+'.data','rb') as file: count_aux = pickle.load(file)
                with open('epu.data','rb') as file: epu = pickle.load(file)
            
                if model[i] == 'HARX':
                    mae_x, mafe_x, fe_ts_x, y_hat_x, std_x = main.har(df=df, f_horizon = f, model = 'lhar-cj+', exog = pd.concat([epu,count_aux],axis=1), regularize = None)
                    
                elif model[i] == 'HARX (AdaLasso)':
                    mae_adalasso, mafe_adalasso, fe_ts_adalasso, betas_adalasso, y_hat_adalasso, std_adalasso = main.har(df=df, f_horizon=f, model='lhar-cj+', exog=pd.concat([epu,count_aux],axis=1), regularize='Adalasso')
                    betas_results.append((stock[j], 'HARX (AdaLasso)', f, betas_adalasso))
            
            # Store results
            mae_results.append((stock[j], model[i], f, mae))      # Mean Absolute Error
            y_hat_results.append((stock[j], model[i], f, y_hat))  # Forecasts
            mafe_results.append((stock[j], model[i], f, mafe))    # Mean Absolute Forecasting Error
            losses.append((stock[j], model[i], f, fe_ts))         # Forecasting Errors
            std_mafe.append((stock[j], model[i], f, std))         # Standard deviation
        
            print("Model {} is done!".format(str(model[i])))
        
        print("Forecasting horizon {}, stock {} done!".format(str(f), str(stock[j])))
  
    
        # Store losses for each model and each stock to later compute the MCS
        db_losses = db_losses + losses 
        losses = np.array(losses).T        
        ll = pd.DataFrame()
        for i in range(np.shape(losses)[1]): # Select the time series component (FE) of the list 
            ll[losses[1,i]]=losses[3,i]
             
            
        # Model Confidence Set
        mcsa = MCS(ll, size=0.10, block_size=150, reps=1000)
        mcsa.compute()
        pvalues = mcsa.pvalues
        included=mcsa.included
        excluded = mcsa.excluded
        mcsa_results.append((stock[j], f, pvalues.index.values, pvalues.values[:,0]))
        
     
        
     
        
# Send to excel for visualization and store results
mae_results = pd.DataFrame(mae_results)
mae_results.to_excel('mae_results.xlsx')

mafe_results = pd.DataFrame(mafe_results)
mafe_results.to_excel('mafe_results.xlsx')

mcsa_results = pd.DataFrame(mcsa_results)
mcsa_results.to_excel('mcsa.xlsx')

std_mafe = pd.DataFrame(std_mafe)
std_mafe.to_excel('std_mafe.xlsx')

with open('betas_results.data','wb') as f:
    pickle.dump(betas_results,f)







# In[2]:


########################
###### Plots ###########
########################

# Plot the Integrated Variance and the Jump Contribution for the entire period

plt.rcParams['figure.dpi'] = 500
plt.rcParams['savefig.dpi'] = 500
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams.update(plt.rcParamsDefault)

bt = []
jc = []
for j in range(len(stock)):
        
    df = main.load_df(stock[j])
    df = df.fillna(method='ffill')
    bt.append((stock[j],df['bt']))
    jc.append((stock[j],df['btv']))
        
index =  jc[0][1].index   
bt_df = pd.DataFrame(1000*np.array([item[1] for item in bt]).T,columns=[item[0] for item in bt],index=index) 
jc_df = pd.DataFrame(10000*np.array([item[1] for item in jc]).T,columns=[item[0] for item in jc],index=index) 


fig = plt.figure()

# Jumps - real economy

plot = jc_df.plot(y=['BOVA11','PETR4','VALE3'],label=(r'$\overline{\mathrm{JC}}$ - Bova',r'$\overline{\mathrm{JC}}$ - Petr',r'$\overline{\mathrm{JC}}$ - Vale'),linewidth=1.3,color=['k','tab:orange','tab:blue'])
plt.ylim([0,12])
plt.xlim([bt_df.index[0], bt_df.index[len(bt_df)-1]])
plt.xticks(rotation=0)
plt.rcParams.update({'font.size': 20})
plt.rcParams['axes.titlepad'] = 18 
plt.title('Jump Contribution',size=23)
plt.legend(loc="upper right",frameon=True,labelspacing=0.10)
plt.xlabel("")
plt.annotate(r'x $10^{-4}$', xy=(165,256), xycoords='figure points', fontsize=14)#,family='Times New Roman')
fig = plot.get_figure()
fig.set_size_inches(18.5, 4)
fig.savefig("Jump Contribution - real_v2.png")
plt.show()

# IV - real economy

plot = bt_df.plot(y=['BOVA11','PETR4','VALE3'],label=(r'$\overline{\mathrm{BP}}$ - Bova',r'$\overline{\mathrm{BP}}$ - Petr',r'$\overline{\mathrm{BP}}$ - Vale'),linewidth=1.3,color=['k','tab:orange','tab:blue'])
plt.ylim([0,4])

plt.xlim([bt_df.index[0], bt_df.index[len(bt_df)-1]])
plt.xticks(rotation=0)
plt.rcParams.update({'font.size': 20})
plt.rcParams['axes.titlepad'] = 18 
plt.title('Integrated Variance',size=23)
plt.legend(loc="upper right",frameon=True,labelspacing=0.10)
plt.xlabel("")
plt.annotate(r'x $10^{-3}$', xy=(165,256), xycoords='figure points', fontsize=14)#,family='Times New Roman')
fig = plot.get_figure()
fig.set_size_inches(18.5, 4)
fig.savefig("Integrated Variance - real_v2.png")
plt.show()

# Jumps - banks

plot = jc_df.plot(y=['BBAS3','BBDC4','ITUB4'],label=(r'$\overline{\mathrm{JC}}$ - Bbas',r'$\overline{\mathrm{JC}}$ - Bbdc',r'$\overline{\mathrm{JC}}$ - Itub'),linewidth=1.3,color=['k','tab:orange','tab:blue'])
plt.ylim([0,12])
plt.xlim([bt_df.index[0], bt_df.index[len(bt_df)-1]])
plt.xticks(rotation=0)
plt.rcParams.update({'font.size': 20})
plt.rcParams['axes.titlepad'] = 18 
plt.title('Jump Contribution',size=23)
plt.legend(loc="upper right",frameon=True,labelspacing=0.10)
plt.xlabel("")
plt.annotate(r'x $10^{-4}$',  xy=(165,256), xycoords='figure points', fontsize=14)#,family='Times New Roman')
fig = plot.get_figure()
fig.set_size_inches(18.5, 4)
fig.savefig("Jump Contribution - banks.png")
plt.show()


# IV - banks

plot = bt_df.plot(y=['BBAS3','BBDC4','ITUB4'],label=(r'$\overline{\mathrm{BP}}$ - Bbas',r'$\overline{\mathrm{BP}}$ - Bbdc',r'$\overline{\mathrm{BP}}$ - Itub'),linewidth=1.3,color=['k','tab:orange','tab:blue'])
plt.ylim([0,4])
plt.xlim([bt_df.index[0], bt_df.index[len(bt_df)-1]])
plt.xticks(rotation=0)
plt.rcParams.update({'font.size': 20})
plt.rcParams['axes.titlepad'] = 18 
plt.title('Integrated Variance',size=23)
plt.legend(loc="upper right",frameon=True,labelspacing=0.10)
plt.xlabel("")
plt.annotate(r'x $10^{-3}$',  xy=(165,256), xycoords='figure points', fontsize=14)#,family='Times New Roman')
fig= plot.get_figure()
fig.set_size_inches(18.5, 4)
fig.savefig("Integrated Variance - banks.png")
plt.show()




# In[3]:

########################
###### Plots ###########
########################


# Plot cross-correlations

stock_title = ['Bova','Petr','Vale','Bbas','Itub','Bbdc']
for j in range(len(stock)):
        
    df = main.load_df(stock[j])
    df = df.fillna(method='ffill')
    mrv = df['mrv']
    btv = df['btv']
    bt = df['bt']
    
    
    returns = pd.read_excel('/Users/muriloandreperespereira/Desktop/Thesis/Vol forecast/rv/Data/returns/'+stock[j]+'_1d.xlsx',index_col='date')
    leverage_pos = pd.Series(returns['returns'])
    leverage_pos[leverage_pos<0]=0
    returns = pd.read_excel('/Users/muriloandreperespereira/Desktop/Thesis/Vol forecast/rv/Data/returns/'+stock[j]+'_1d.xlsx',index_col='date')
    leverage_neg = pd.Series(returns['returns'])
    leverage_neg[leverage_neg>0]=0
    leverage_neg = abs(leverage_neg)
    
    
    mrv = np.log(mrv)
    bt = np.log(bt)
    btv = np.log(1+btv*10000)
    
    
    corr_bt_list = []
    corr_btv_list = []
    corr_pos_list = []
    corr_neg_list = []
    
    #contemporaneous corr
    corr_bt = np.corrcoef(mrv,bt)
    corr_btv = np.corrcoef(mrv,btv)
    corr_pos = np.corrcoef(mrv,leverage_pos)
    corr_neg = np.corrcoef(mrv,leverage_neg)
        
    corr_bt_list.append(corr_bt[0,1])
    corr_btv_list.append(corr_btv[0,1])
    corr_pos_list.append(corr_pos[0,1]) 
    corr_neg_list.append(corr_neg[0,1])
    
    l= 20
    for i in range(1,l):
        corr_bt = np.corrcoef(mrv[i:],bt[:-i])
        corr_btv = np.corrcoef(mrv[i:],btv[:-i])
        corr_pos = np.corrcoef(mrv[i:],leverage_pos[:-i])
        corr_neg = np.corrcoef(mrv[i:],leverage_neg[:-i])
        
        corr_bt_list.append(corr_bt[0,1])
        corr_btv_list.append(corr_btv[0,1])
        corr_pos_list.append(corr_pos[0,1]) 
        corr_neg_list.append(corr_neg[0,1])

   
    X = pd.DataFrame(np.array((corr_bt_list,corr_btv_list,corr_pos_list,corr_neg_list)).T)
   
    plot = X.plot(y=[0,1,2,3],label=(r'$\overline{BP}$',r'$\overline{JC}$','Leverage(+)','Leverage(-)'),linewidth=1.1,fillstyle='none',style=['-','-','-','--'],marker='.',color=['tab:orange','tab:blue','dimgrey','k'])
    plt.xticks(rotation=0)
    plt.ylim([-0.01,1])
    plt.xlim([0, l])
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['axes.titlepad'] = 13
    plt.title(stock_title[j],size=23)
    plt.legend(loc="upper right",frameon=True,labelspacing=0.10)
    plt.xlabel("Lags",fontsize=15.5)
    fig= plot.get_figure()
    fig.set_size_inches(7, 5)
    fig.savefig(stock[j])
    plt.show()



   
    
   

# In[4]:

###################################
###### Descriptive Stat ###########
###################################
    

pd.options.display.float_format = '{:.5f}'.format
stat=[]

for j in range(len(stock)):
        
    df = main.load_df(stock[j])
    df = df.fillna(method='ffill')
    print(stats.skew(df['mrv']),stats.kurtosis(df['mrv']))
    print(df['mrv'].describe()) 
    stat.append((stock[j],df['mrv'].describe()))

stat = pd.DataFrame(np.array(stat).T)
stat.to_excel('stat.xlsx')
ADF(df['mrv'])
    



# Descriptive stats for jumps
stat=[]
stat_asymmetries = []
for j in range(len(stock)):
        
    df = main.load_df(stock[j])*10000
    df = df.fillna(method='ffill')
    returns = pd.read_excel('/Users/muriloandreperespereira/Desktop/Thesis/Vol forecast/rv/Data/returns/'+stock[j]+'_1d.xlsx',index_col='date')
   
    jc = pd.concat([df['btv'],returns],axis=1)
    jc['jc+'] = 0
    jc['jc-'] = 0
    
    for i in range(len(jc)):
        if jc.iloc[i,1]>0:
            jc.iloc[i,2]= jc.iloc[i,0]
        else:
            jc.iloc[i,3]=jc.iloc[i,0]
    
    jc = jc.replace(0, np.nan)
    mean_jc_plus = np.mean( jc.dropna(subset=['jc+']).iloc[:,2]) 
    mean_jc_minus = np.mean( jc.dropna(subset=['jc-']).iloc[:,3]) 
    print(stock[j],mean_jc_plus,mean_jc_minus)
    stat_asymmetries.append((stock[j],mean_jc_plus,mean_jc_minus))

np.mean(df['btv']/df['mrv'])
stat = pd.DataFrame(np.array(stat).T)
stat.to_excel('stat.xlsx')




# In[4]:

###################################
###### Betas Adalasso ###########
###################################

with open('betas.data','rb') as f: # Open betas computed before
          betas = pickle.load(f)

results = []
for k in range(len(betas)):
    df = pd.DataFrame(betas[k][3])
    df = df.drop(0,axis=1)
    
    for i in range(len(df.columns)):
        count=0
        aux = df.iloc[:,i]
        for j in range(len(aux)):
            if aux[j] != 0:
                count = count+1
        
        p = count/442 
        results.append((k,betas[k][2],i,p))
        
            
prop= np.zeros((16,6))
for i in range(len(results)):
    if results[i][1]==22:
        prop[int(results[i][2]),int(results[i][0])-18]=results[i][3]
        
prop = pd.DataFrame(prop)        
prop.to_excel('prop.xlsx')
    


