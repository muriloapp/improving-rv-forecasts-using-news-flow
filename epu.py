#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 23:05:24 2021

@author: muriloandreperespereira


Computes the Economic Policy Uncertainty Index for Brazil based on Baker, Bloom and Davis (2016). 

"""


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import pickle
from datetime import timedelta 


# In[1]:
    
def find_news_epu(newspaper):
        " Find news using specific keywords."
    
        Corpus = [] 
        keywords1 = [ "economia" , "Economia" , "econômico" , "Econômico" , "incerteza" , "Incerteza" , "incerto" , "Incerto" , "inesperado" , "Inesperado" , "surpresa" , "Surpresa" ]
        keywords2 = [ "congresso", "Congresso", "senado" , "Senado" , "défict" , "Déficit" , "Banco Central" , "legislação" , "Legislação" , "orçamento" , "Orçamento" , "Imposto" , "imposto" , "alvorada" , "Alvorada" , "planalto" , "Planalto" , "Câmara" , "câmara" , " Lei " , " lei " , "tarifa" , "Tarifa", "Impeachment" , "impeachment" , " cpi " , " CPI " ]
        
        for i in range(len(newspaper)):
            if any(word in newspaper[i][2] for word in keywords1):
                if any(word in newspaper[i][2] for word in keywords2):
                        Corpus.append(newspaper[i])
        
        return Corpus
    
    
def preprocess_epu(newspaper):
        " Preprocess Corpus."
        
        # Transform to datetime
        date = [item[0] for item in newspaper]
        for i in range(len(date)):
            date[i] = datetime.strptime(str(date[i]), '%d/%m/%Y').date()
        
        # DataFrame
        dic = {'Date':date ,'Count': 1}     
        df = pd.DataFrame(data=dic)
        # Group by date        
        df = df.groupby(['Date'])['Count'].agg('sum')
        df = pd.DataFrame(df)
        std_ma = np.std(df.iloc[:,0])
        
        # Normalize
        df['normalized'] = 0
        df.iloc[:,1] = (df.iloc[:,0])/std_ma
        df = pd.DataFrame(df.iloc[:,1])
       
        return df
    
  
def compute_epu(Estadao,Folha,Valor):
        "The entire procedure is described in BBK (2016)."
    
        Estadao = find_news_epu(Estadao)
        Folha = find_news_epu(Folha)
        Valor = find_news_epu(Valor)
        
        df_Estadao = preprocess_epu(Estadao)
        df_Folha = preprocess_epu(Folha)
        df_Valor = preprocess_epu(Valor)
        
        df = pd.concat([df_Estadao,df_Folha,df_Valor],axis=1)
        df = df.sort_index(axis=1)
        df = df.fillna(0)
        
        # Considering all newspapers
        df['epu'] = 0
        for i in range(len(df)):
            df.iloc[i,3] = (df.iloc[i,0]+df.iloc[i,1]+df.iloc[i,2])/3
        df = pd.DataFrame(df.iloc[:,3])
        df = df.sort_index()
        mean_z = np.sum(df)/len(df)
        #df = df*100/mean_z
        df = df/mean_z
        
        
        # Importat: match trading days
        dates_list = pd.read_excel('/Users/muriloandreperespereira/Desktop/Thesis/Vol forecast/rv/date.xlsx')
        for i in range(len(dates_list)):
            dates_list.iloc[i,0] = datetime.strptime(str(dates_list.iloc[i,0]), '%Y%m%d').date()
        dates_list = list(dates_list.values[:,0])
        
        
        # Moving Averages
        epu = pd.DataFrame(index=dates_list,columns=['epu_1'])
        epu = epu.fillna(0)
        for i in range(1,len(epu)):
            init =  np.where(df.index >= datetime.strptime(str(epu.index[i-1]),'%Y-%m-%d').date())[0].min()
            end = np.where(df.index >= datetime.strptime(str(epu.index[i]),'%Y-%m-%d').date())[0].min()
            
            epu.iloc[i,0] = np.mean(df.iloc[init+1:end+1,0])
        
        # Five days MA
        epu['epu_5'] = 0
        for i in range(4,len(epu)):
            epu.iloc[i,1] = np.mean(epu.iloc[i-4:i+1,0])
        
        # Twenty-two days MA
        epu['epu_22'] = 0
        for i in range(21,len(epu)):
            epu.iloc[i,2] = np.mean(epu.iloc[i-21:i+1,0])
        
        epu = epu.iloc[21:,:]

        return epu
    

# Load raw database
with open('/Users/muriloandreperespereira/Desktop/Thesis/base de notícias/raw data/Estadão.txt','rb') as f:
          Estadao = pickle.load(f)
with open('/Users/muriloandreperespereira/Desktop/Thesis/base de notícias/raw data/Folha.txt','rb') as f:
          Folha = pickle.load(f)
with open('/Users/muriloandreperespereira/Desktop/Thesis/base de notícias/raw data/Valor.txt','rb') as f:
          Valor = pickle.load(f)
    
# Run and store
epu = compute_epu(Estadao,Folha,Valor)
# with open('epu.data','rb') as f:
#           epu = pickle.load(f)

   


# In[2]:


## Plot EPU

# plt.rcParams['figure.dpi'] = 500
# plt.rcParams['savefig.dpi'] = 500
# #plt.rcParams["font.family"] = "Times New Roman"


plot = epu.plot(y=['epu_22'],linewidth=2.5,color=['tab:blue'])
plt.ylim([60,180])
plt.xticks(rotation=0)
plt.xlim([epu.index[0], epu.index[len(epu)-15]])
plt.rcParams.update({'font.size': 20})
plt.rcParams['axes.titlepad'] = 18
plt.title('EPU Index',size=23)
plt.legend('',frameon=False)
plt.xlabel("")

plt.annotate('Lower House\nImpeaches\nRoussef', xy=(180,265), xycoords='figure points', fontsize=17)#, fontsize=13,family='Times New Roman')
plt.annotate('Tainted-Meat\nScandal', xy=(480,245), xycoords='figure points', fontsize=17),#family='Times New Roman')
plt.annotate('Lorry\nDrivers Strike', xy=(745,260), xycoords='figure points', fontsize=17)#,family='Times New Roman')
plt.annotate('Elections', xy=(853,142), xycoords='figure points', fontsize=15)#,family='Times New Roman') #28/ot
plt.annotate('Concerns about\nthe Pension Reform', xy=(980,324), xycoords='figure points', fontsize=17)#,family='Times New Roman')

fig= plot.get_figure()
fig.set_size_inches(18.5, 6)
fig.savefig("EPU_v2.png")
plt.show()




    
    
    

