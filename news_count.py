#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 23:05:24 2021

@author: muriloandreperespereira


" Computes firm specific news based on Fernandes and Pereira (2022) "

"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import pickle
from datetime import timedelta 
import os

    
    

def find_news(Corpus_raw,keyword_1,keyword_2,c,keyword_3=None):  
    " Find news based on specific keywords. "
    
    blogs = ['Blogs','Blogs','blog','blogs','poder'] # Remove
    selector = ['lucro','Lucro','receita','Receita','despesa','Despesa','prejuízo','Prejuízo','prejuizo'] # Requirement
    Corpus = []
    
    # Select news based on pre-specified keywords
    for i in range(0,len(Corpus_raw)):
        
        #*** Choose keyword
        keyword1 = str(keyword_1)
        keyword2 = str(keyword_2)
        
        if keyword_3 != None:
            keyword3 = str(keyword_3)
        
        
        count=c
        if Corpus_raw[i][2].count(keyword1) >= count:

                     if any(key in Corpus_raw[i][2] for key in selector):
                        if any(word in Corpus_raw[i][1] for word in blogs):
                            pass
                        else:                
                            Corpus.append(Corpus_raw[i][:])

        elif Corpus_raw[i][2].count(keyword2) >= count:
                    if any(key in Corpus_raw[i][2] for key in selector):
                        if any(word in Corpus_raw[i][1] for word in blogs):
                            pass
                        else:                 
                            Corpus.append(Corpus_raw[i][:])
        
        elif keyword_3 != None and Corpus_raw[i][2].count(keyword3) >= count:  
                    if any(key in Corpus_raw[i][2] for key in selector):
                        if any(word in Corpus_raw[i][1] for word in blogs):
                            pass
                        else:              
                            Corpus.append(Corpus_raw[i][:])
    
    return Corpus


def count_news_fun(newspaper):
    " Count news and separate open market and overnight based on datetime. "
    

    date = [item[0] for item in newspaper]
    time = [item[3] for item in newspaper]
    
    for i in range(len(date)):
        date[i] = datetime.strptime(str(date[i]), '%d/%m/%Y').date()
   
    # DataFrame
    dic = {'date':date ,'time':time,'count': 1}     
    df = pd.DataFrame(data=dic)
    
    df = df.sort_values(['date']) 
    df = df.set_index(['date'])
    index = df.index
    
    
    # Splitting news
    df['open_market+1'] = 0
    df['overnight']=0
    df['overnight+1'] = 0
    
    # Open market between 10am and 6pm
    for i in range(len(index)):
        if df.iloc[i,0].hour >= 10 and df.iloc[i,0].hour <18:
             df.iloc[i,2] = 1
        elif df.iloc[i,0].hour < 10:
            df.iloc[i,3] = 1
        elif df.iloc[i,0].hour >= 18:
            df.iloc[i,4] = 1
            
    
    # Searching for trading days to adjust datetime
    dates_list = pd.read_excel('/Users/muriloandreperespereira/Desktop/Thesis/Vol forecast/rv/date.xlsx')
    for i in range(len(dates_list)):
            dates_list.iloc[i,0] = datetime.strptime(str(dates_list.iloc[i,0]), '%Y%m%d').date()
    dates_list = list(dates_list.values[:,0])

    
    # Adjusting overnight data
    count = pd.DataFrame(index=dates_list,columns=['overnight'])  
    count = count.fillna(0)
    aux1 = df.groupby(['date'])['overnight'].agg('sum')

    for i in range(1,len(count)):
            init =  np.where(aux1.index >= datetime.strptime(str(count.index[i-1]),'%Y-%m-%d').date())[0].min()
            end = np.where(aux1.index >= datetime.strptime(str(count.index[i]),'%Y-%m-%d').date())[0].min()
            
            count.iloc[i,0] = np.mean(aux1[init+1:end+1])


    count['overnight2']=0
    aux2 = df.groupby(['date'])['overnight+1'].agg('sum')

    for i in range(1,len(count)):
            init =  np.where(aux2.index >= datetime.strptime(str(count.index[i-1]),'%Y-%m-%d').date())[0].min()
            end = np.where(aux2.index >= datetime.strptime(str(count.index[i]),'%Y-%m-%d').date())[0].min()
            
            count.iloc[i,1] = np.mean(aux2[init+1:end+1])
            if count.iloc[i,1] == np.nan:
                count.iloc[i,1] = np.mean(aux2[init:end+1])
                

    count['open']=0  
    aux2 = df.groupby(['date'])['open_market+1'].agg('sum')
   
    for i in range(1,len(count)):
            init =  np.where(aux2.index >= datetime.strptime(str(count.index[i-1]),'%Y-%m-%d').date())[0].min()
            end = np.where(aux2.index >= datetime.strptime(str(count.index[i]),'%Y-%m-%d').date())[0].min()
            
            count.iloc[i,2] = np.mean(aux2[init+1:end+1])
            if count.iloc[i,2] == np.nan:
                count.iloc[i,2] = np.mean(aux2[init:end+1])  

    count = count.fillna(0)
    count['overnight'] = count['overnight']+count['overnight2']
    count = count.drop(['overnight2'],axis=1)

    


    # Including lagged news based on Corsi (2009) ideas
    count['d_1']=count['overnight']+count['open']

    
    count['d_5'] = 0
    for i in range(4,len(count)):
            count.iloc[i,3] = np.mean(count.iloc[i-4:i+1,2])

    count['d_22'] = 0
    for i in range(21,len(count)):
            count.iloc[i,4] = np.mean(count.iloc[i-21:i+1,2])
        
    count = count.iloc[21:,:]
    
    return count


def count_news():
    
    os.chdir('/Users/muriloandreperespereira/Tese/News')
        
    ## Load news. This is a raw database
    with open('/Users/muriloandreperespereira/Downloads/Estadao.data','rb') as f:
         Estadao = pickle.load(f)
    with open('/Users/muriloandreperespereira/Downloads/Folha.data','rb') as f:
         Folha = pickle.load(f)
    with open('/Users/muriloandreperespereira/Downloads/Valor.data','rb') as f:
        Valor = pickle.load(f)
    Corpus = Folha + Valor + Estadao
    
    
    n=1
    # Find news
    Corpus_PETR4 = find_news(Corpus, 'Petrobras','Petrobrás',1)
    Corpus_VALE3 = find_news(Corpus, ' Vale ','VALE3',n)
    Corpus_ITUB4 = find_news(Corpus, 'Itaú','Itau',n)
    Corpus_BBDC4 = find_news(Corpus, 'Bradesco','bradesco',n)
    Corpus_BBAS3 = find_news(Corpus, 'Banco do Brasil','banco do Brasil',n)
    Corpus_ABEV3 = find_news(Corpus, 'Ambev','AMBEV',n)

    # Find news
    count_PETR4 = count_news_fun(Corpus_PETR4)
    count_VALE3 = count_news_fun(Corpus_VALE3)
    count_ITUB4 = count_news_fun(Corpus_ITUB4)
    count_BBDC4 = count_news_fun(Corpus_BBDC4)
    count_BBAS3 = count_news_fun(Corpus_BBAS3)
    count_ABEV3 = count_news_fun(Corpus_ABEV3)
    
    
    return count_PETR4,count_VALE3,count_ITUB4,count_BBDC4,count_BBAS3,count_ABEV3


count_PETR4,count_VALE3,count_ITUB4,count_BBDC4,count_BBAS3,count_ABEV3 = count_news()



# In[]:


count_PETR4 = count_PETR4.drop(['d_1'],axis=1)
count_VALE3 = count_VALE3.drop(['d_1'],axis=1)
count_ITUB4 = count_ITUB4.drop(['d_1'],axis=1)
count_BBDC4 = count_BBDC4.drop(['d_1'],axis=1)
count_BBAS3 = count_BBAS3.drop(['d_1'],axis=1)
count_ABEV3 = count_ABEV3.drop(['d_1'],axis=1)


count_PETR4 = count_PETR4.drop(['open'],axis=1)
count_VALE3 = count_VALE3.drop(['open'],axis=1)
count_ITUB4 = count_ITUB4.drop(['open'],axis=1)
count_BBDC4 = count_BBDC4.drop(['open'],axis=1)
count_BBAS3 = count_BBAS3.drop(['open'],axis=1)
count_ABEV3 = count_ABEV3.drop(['open'],axis=1)


with open('count_BOVA11.data','rb') as f:
          count_BOVA11 = pickle.load(f)





