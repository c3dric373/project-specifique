#!/usr/bin/env python
# coding: utf-8

# ## SetUp directories

# In[1]:


import os
data_directory = '../data/'
if(not os.path.exists((data_directory))):
     os.makedirs(data_directory)


# # Preprocessing

# ## SetUp Parameters

# In[2]:




corpus_file = 'corpus_check_long.csv'
corpus_path = data_directory + corpus_file
# We will create a temporary file with the results of the preprocessing this file will be deleted after 
#the execution of the script
temp_file_eval = "../data/evalFile.txt"

# File Name where we will store the training data 
train_path = data_directory + 'trainFile.txt'

# File name where we will store the evaluation data
eval_file = data_directory + 'eval.csv'

# Prediction File 
prediction_file = data_directory + 'prediction.csv'

# Name of the column storing the article 
article = 'corpus'

utilities_path = '../utilities/'# DataSetPath 
prediction_path = utilities_path + 'groupC_scrap.obj'
prediction_csv_path = utilities_path + 'prediction.csv'


# ## Load Datasets to preprocess

# In[3]:


import pickle
import pandas as pd
file = open(prediction_path, 'rb') 
df_prediction = pd.DataFrame(pickle.load(file))
df = pd.read_csv(corpus_path)


# In[4]:


def get_corrupt_data(df):
    indexNames = []
    for counter,data in enumerate(df.iterrows()):
        i, row = data
        tmp = df.corpus[i]
        if (("�") in tmp) or (len(tmp.split())<50):
            indexNames.append(i)
    return indexNames
    


# In[5]:


def filter_dataframe(dataframe):
    # Remove corrupt Data and filter articles that have less than 50 words
    indexNames = get_corrupt_data(dataframe)  
    dataframe.drop(indexNames , inplace=True)
    
    # Filter companies that have at least 7 articles
    top = dataframe["siren"].value_counts()
    top = top.where(top>=7).dropna()
    topList = list(top.index)
    dataframe = dataframe[dataframe["siren"].isin(topList)]
    
    # Filter articles longer than 1,000,000 characters
    dataframe = dataframe[dataframe[article].astype(str).map(len)<1000000]
    
    return dataframe
    


# In[6]:


import re
import string
from nltk.tokenize import sent_tokenize 
def cleaning(doc):
    doc = doc.replace('\n', ' ')
    doc = doc.replace('\r\n', ' ')
    doc = doc.replace('\r', ' ')
    doc = doc.replace('\t', ' ')
    return doc 
def remove_numbers(doc):
    doc = re.sub("\d+", "", doc)
    doc = doc.replace('m€', '')
    doc = doc.replace('k€', '')   
    return doc
# Tokenize text
def preprocessing(doc,train=False):
        # Translator used to remove punctuation
        translator = str.maketrans(' ', ' ', string.punctuation)

        # Remove «»
        doc = doc.replace("«", " ")
        doc = doc.replace("»", " ")

        # To lowercase 
        doc = doc.lower()
        
        # Remove url's
        doc = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', doc, flags=re.MULTILINE)
        
        # Cleaning
        doc = cleaning(doc)
        
        # Remove numbers
        doc = remove_numbers(doc)
        
    
        # Remove multiple wite spaces 
        doc = re.sub(' +', ' ',doc)
        
        # Remove unicode breaking character
        doc = doc.replace(u'\xa0', u' ')
        
        if train: 
            result = []
            sentences = sent_tokenize(doc)
            for sent in sentences: 
                   # Remove punctuation
                sent = sent.translate(translator)
                sent += "\n"
                result.append(sent)
            return "".join(result)
        else:
            doc += "\n"
            return doc 


# In[7]:


def read_file(path):
    with open(path) as f:
        content = f.readlines()
    return content
    


# In[8]:


def merge_file(temp_file,path):
    subprocess.check_output(["cat " + temp_file + "*" + ' > ' + path],shell=True)
    subprocess.check_output(["rm " + temp_file + "*"],shell=True)


# In[9]:


def preprocess_and_write_to_file(dataframe,fileName='data',train=False,index=0):
    total_len = (len(dataframe))
    third = int(len(dataframe)/3)
    f = codecs.open(fileName + str(index) + '.txt' , 'w', 'utf-8')
    for counter,data in enumerate(dataframe.iterrows()):
        i, row = data
        if(counter%third==0):
            print("Thread " + str(index) + " processed " + str(counter) + "/" + str(total_len))
        preprocessed_text = preprocessing((row[article]),train)
        f.write(preprocessed_text)  # python will convert \n to os.linesep
    f.close()  


# In[10]:


import codecs
import multiprocessing
import numpy as np
import time
import subprocess
def multi_thread_preprocessing(dataframe,path,train=True,threads=3):
    temp_file_name = "tmp_"
    chunks = np.array_split(dataframe,threads)
    manager = multiprocessing.Manager()
    threads = []
    for index,chunk in enumerate(chunks):
        thread = multiprocessing.Process(target=preprocess_and_write_to_file, args=(chunk,temp_file_name,train,index))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if(train):
        merge_file(temp_file_name,path)
    else:
        new_df = dataframe.copy()
        temp_file_eval = 'eval_file_tmp'
        merge_file(temp_file_name,temp_file_eval)
        data = read_file(temp_file_eval)
        print(len(data))
        subprocess.run(["rm", temp_file_eval])
        new_df['corpus'] = data
        new_df.to_csv(path)


# ### Filter

# In[11]:


df = filter_dataframe(df)
df_prediction = filter_dataframe(df_prediction)


# ### Preprocessing

# In[197]:


multi_thread_preprocessing(df_prediction,prediction_file,train=False)


# In[198]:


multi_thread_preprocessing(df,eval_file,train=False)


# In[199]:


multi_thread_preprocessing(df,train_path,train=True)


# In[ ]:




