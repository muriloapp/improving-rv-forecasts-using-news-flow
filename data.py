#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:38:45 2021

@author: muriloandreperespereira


" Get data from Azure"
"""

#import os, uuid

# from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

# try:
#     print("Azure Blob Storage v" + __version__ + " - Python quickstart sample")
    
#     connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

#     # Quick start code goes here
# except Exception as ex:
#     print('Exception:')
#     print(ex)


# STORAGEACCOUNTURL = "https://murilopereira.blob.core.windows.net"
# STORAGEACCOUNTKEY = "DYO6/oja/JHXb43QmVhD/Rb7yhJaGYzoBYZQpWpuiVyWBdWxmUyoKC0M2jViiLWmfz8cjtI2Lo0SWN7ndOV39w=="
# CONTAINERNAME = "data"
# BLOBNAME = "NEG/20160301/PETR4.zip"

# blob_service_client_instance = BlobServiceClient(
#     account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)

# blob_client_instance = blob_service_client_instance.get_blob_client(
#     CONTAINERNAME, BLOBNAME, snapshot=None)

# blob_data = blob_client_instance.download_blob()

# data = blob_data.readall()



from azure.storage.blob import BlobServiceClient

STORAGEACCOUNTURL = "https://murilopereira.blob.core.windows.net"
STORAGEACCOUNTKEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
container_name = "data"
date = str(20160301)
stock = 'PETR4'
blobname = 'NEG/' + date + '/' + stock + '.zip'


blob_service_client= BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)

# Download blob to local zip file

local_file_path = blobname
with open(date+'.zip', "wb") as download_file:
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blobname)
    download_file.write(blob_client.download_blob().readall())


from zipfile import ZipFile

# Open zipped files

with ZipFile('/Users/muriloandreperespereira/Desktop/Thesis/Vol forecast/rv/Data/PETR4/'+date+'.zip') as myzip:
    with myzip.open(stock+'.txt') as myfile:
        aux = myfile.read()


import pandas as pd
import io

buf_bytes = io.BytesIO(aux)
df = pd.read_csv(buf_bytes,sep=";", header=None)


