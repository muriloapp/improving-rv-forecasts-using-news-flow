#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:09:52 2021

@author: muriloandreperespereira

Access news articles from Estadão, Folha and Valor to extract information.

"""


from datetime import datetime
from selenium import webdriver
import pickle
    

def search_time(Corpus, newspaper):
    "Search time of each news included in Corpus database using Google Chrome"
    
    problem = [] # Store eventual problems
    
    driver = webdriver.Chrome(executable_path="/Users/muriloandreperespereira/chromedriver-2")
    
    for i in range(len(Corpus)):
        
        try:
            url = Corpus[i][1]
            driver.get(url)
            
            if newspaper == 'Folha':
                try:
                    p_element = driver.find_element_by_xpath("//*[@id='c-news']/div[5]/div/div/div/div/div[1]/div/div[1]/time")  
                    Corpus[i].append(p_element.text)
                except Exception:
                    p_element = driver.find_element_by_xpath("//*[@id='news']/header/time")
                    Corpus[i].append(p_element.text)
            
            
            elif newspaper == 'Estadao':
                try:
                    p_element = driver.find_element_by_class_name("n--noticia__state-desc")  
                    Corpus[i].append(p_element.text)
                except:
                    p_element = driver.find_element_by_class_name("n--noticia__state")
                    Corpus[i].append(p_element.text)
                
            elif newspaper == 'Valor':
                p_element = driver.find_element_by_class_name("content-publication-data__updated")
                Corpus[i].append(p_element.text)
        
        except Exception:
            problem.append(i)
            pass
    
    return Corpus





## Run

# Valort = search_time(Corpus, 'Valor')
# Folhat = search_time(Corpus, 'Folha')
# Estadaot = search_time(Corpus, 'Estadao')


## Save Corpus with datetime

#with open('Valor.data','wb') as f: pickle.dump(Valort,f)
#with open('Folha.data','wb') as f: pickle.dump(Folhat,f)
#with open('Estadao.data','wb') as f: pickle.dump(Estadaot,f)


with open('Estadaot.data', 'rb') as f:
    Estadao = pickle.load(f)
with open('Valort.data', 'rb') as f:
    Valor = pickle.load(f)
with open('Folhat.data', 'rb') as f:
    Folha = pickle.load(f)


# In[1]:


## Minor adjustments to remove/include words and datetime module


# Removing the word "atualizado"
# keywords1 = ['atualizado','Atualizado']
Folhat=[]
removed=[]
for i in range(3,len(Folha)):
    txt = Folha[i][2][0:80]
    txt_1 = Folha[i-1][2][0:80]
    txt_2 = Folha[i-2][2][0:80]
    txt_3 = Folha[i-3][2][0:80]
    if txt == txt_1:
        removed.append(Folha[i])
    elif txt == txt_2:
        removed.append(Folha[i])
    elif txt == txt_2:
        removed.append(Folha[i])
    else:
        Folhat.append(Folha[i])



## Estadao
for i in range(0,len(Estadao)):
        news_time = Estadao[i][3]
        Estadao[i][3] = news_time[(news_time.find("|")+2):]
        if Estadao[i][3] == '0':
            Estadao[i][3] ='00h00'
        
        try:
            try:
                try:
                    Estadao[i][3] = datetime.strptime(Estadao[i][3], '%Hh%M').time()
                except Exception:
                    news_time = Estadao[i][3]
                    Estadao[i][3] = news_time[(news_time.find("|")+2):]
                    Estadao[i][3] = datetime.strptime(Estadao[i][3], '%Hh%M').time()
            except Exception:
                    Estadao[i][3] = datetime.strptime(Estadao[i][3], '%Hh').time()
        except Exception:
                news_time = Estadao[i][3]
                Estadao[i][3] = news_time[:(news_time.find("Atualizado")-1)]
                try:    
    
                    Estadao[i][3] = datetime.strptime(Estadao[i][3], '%Hh%M').time()
                except Exception:
                     news_time = Estadao[i][3]
                     Estadao[i][3] = news_time[:(news_time.find("Correções")-1)]
                     Estadao[i][3] = datetime.strptime(Estadao[i][3], '%Hh%M').time()
                
                           
                
## Valor/Folha
for i in range(len(Folhat)):
        news_time = Folhat[i][3]
        Folhat[i][3] = news_time[(news_time.find("h")-2):]
        if  Folhat[i][3][0] ==' ':
            Folhat[i][3] =  Folhat[i][3][0].replace(' ','0')+ Folhat[i][3][1:]
        if 'Atualizado' in  Folhat[i][3]:
             Folhat[i][3] = news_time[(news_time.find("h")-2):(news_time.find("h")+3)]
        if 'Erramos' in  Folhat[i][3]:
             Folhat[i][3] = news_time[(news_time.find("h")-2):(news_time.find("h")+3)]
            
        # Transforming to datetime
        Folhat[i][3] = datetime.strptime(Folhat[i][3], '%Hh%M').time()
        


