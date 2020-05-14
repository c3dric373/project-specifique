#!/usr/bin/env python
# coding: utf-8

# ## SetUp directories

# In[58]:


import os
data_directory = '../data/'
if(not os.path.exists((data_directory))):
     os.makedirs(data_directory)


# ## SetUp Parameters

# In[59]:


corpus_file = 'corpus_check_long.csv'
corpus_path = data_directory + corpus_file
# We will create a temporary file with the results of the preprocessing this file will be deleted after 
#the execution of the script
temp_file_eval = "../data/evalFile.txt"

# File Name where we will store the training data 
train_path = data_directory + 'trainFile.txt'

# File name where we will store the evaluation data
eval_file = data_directory + 'eval.csv'

# Name of the column storing the article 
article = 'corpus'


# # Preprocessing

# In[60]:


import pandas as pd
df = pd.read_csv(corpus_path)


# In[61]:


df


# In[62]:


indexNames = []
def get_corrupt_data(df):
    for counter,data in enumerate(df.iterrows()):
        i, row = data
        tmp = df.corpus[i]
        if ("�") in tmp:
            indexNames.append(i)
get_corrupt_data(df)      
df.drop(indexNames , inplace=True)


# ## Filter : keep only companies that have at least 7 articles, and their 
# 

# In[63]:


#Build list of companies that have more then 7 articles in the corpora
top = df["siren"].value_counts()
top = top.where(top>=7).dropna()
topList = list(top.index)
df = df[df["siren"].isin(topList)]


# ## Filter: discard articles that are longer than 1,000,000 characters

# In[64]:


df = df[df[article].astype(str).map(len)<1000000]
df


# ## Filter: discard articles that are longer than 100 words
# 

# In[65]:


import re
import contractions
import string
from nltk.tokenize import sent_tokenize 

translator = str.maketrans(' ', ' ', string.punctuation)


# In[66]:


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


# In[67]:


temp_train_name = 'dataTrain'
temp_eval_name = 'dataEval'
# Tokenize text
def preprocessing(doc,train=False):        
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

def preprocess_and_write_to_file(dataframe,train,index=0):
    if(train):
        fileName = temp_train_name
    else:
        fileName = temp_eval_name
    f = codecs.open(fileName + str(index) + '.txt' , 'w', 'utf-8')
    for counter,data in enumerate(dataframe.iterrows()):
        i, row = data
        if(counter%5000==0):
            print("Thread " + str(index) + "processed " + str(counter) + "/" + str(dataframe.count()))
        preprocessed_text = preprocessing((row[article]),train)
        f.write(preprocessed_text)  # python will convert \n to os.linesep
    f.close()  


# We want to create two files: one for training which will consist of each sentence of each document per line and 
# an eval file which will be in csv format containing the name of the article, the url it originated from and the 
# preprocessed article itself.

# In[68]:


train_ = [True,False]
import codecs
import multiprocessing
import numpy as np
chunks = np.array_split(df,3)
manager = multiprocessing.Manager()
threads = []
for train in train_:
    for index,chunk in enumerate(chunks):
        thread = multiprocessing.Process(target=preprocess_and_write_to_file, args=(chunk,train,index))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


# In[69]:


import subprocess
subprocess.check_output(["cat " + temp_train_name + "*.txt" + ' > ' + train_path],shell=True)
subprocess.check_output(["cat " + temp_eval_name + "*.txt" + ' > ' + temp_file_eval],shell=True)
subprocess.check_output(["rm data*.txt"],shell=True)


# Creating the Eval csv file

# In[70]:


def read_file(path):
    with open(path) as f:
        content = f.readlines()
    return content
    


# In[73]:


data = read_file(temp_file_eval)


# In[74]:


subprocess.run(["rm", temp_file_eval])


# In[75]:


df['preprocessedCorpus'] = data
del df['id']
del df['corpus']
df = df.rename({'preprocessedCorpus': article}, axis='columns')
df.to_csv(eval_file)


# In[76]:


df


# In[ ]:




