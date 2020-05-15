#!/usr/bin/env python
# coding: utf-8

# # Train

# ## SetUp directories

# In[1]:


print('Starting Training')
import os
# Location of models
model_directory = '../models'
if(not os.path.exists(model_directory)):
    os.makedirs(model_directory)


# ## Parameters to set: very important!
# ### Please set the glove Path correctly
# Adjust Path to your local GloVe Repo
# https://github.com/stanfordnlp/GloVe

# In[2]:


# Path to utilities
utilities_path = '../utilities'

# Path of your local glove directory
glove_path = '../utilities/GloVe/'        
train_path = '../data/trainFile.txt'

#Glove Script location
utility_glove_script = '../utilities/demo.sh'

# Dimension of vectors
dim_vec = 300

# Number of threads used during training, should be equal to number of cores if one wants to minimize training time 
threads = 4


# In[3]:


from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y")
global_path = "../models/"
bin_path = ".bin"
model_path = ".model"
txt_path = ".txt"
path_ft_bin =  global_path + "ft_"+ dt_string + bin_path
path_ft_txt =  global_path + "ft_"+ dt_string + txt_path
path_w2v =  global_path + "w2v_"+ dt_string + model_path
model_path_glove = global_path + "gloVe_"+ dt_string + txt_path


# ## Fasttext

# In[6]:


import fasttext
print('Start Trainin FastText')
model = fasttext.train_unsupervised(train_path,thread=threads,epoch=9,dim=dim_vec)


# In[11]:


model.save_model(path_ft_bin)


# We only want to save the vectors at it will cost less storage space

# In[12]:


from fasttext import load_model

# original BIN model loading
f = load_model(path_ft_bin)
lines=[]

# get all words from model
words = f.get_words()

with open(path_ft_txt,'w') as file_out:
    # the first line must contain number of total words and vector dimension
    file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")

    # line by line, you append vectors to VEC file
    for w in words:
        v = f.get_word_vector(w)
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)
        try:
            file_out.write(w + vstr+'\n')
        except:
            pass


# In[49]:


import subprocess
subprocess.run(["rm", path_ft_bin])


# ## Word2Vec

# In[13]:


print('Start Trainin w2vec')
with open(train_path) as f:
    corpus = f.readlines()
res = []
for sent in corpus: 
    sent = sent[0:len(sent)-1]
    res.append(sent.split(" "))


# In[16]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models import Word2Vec
model = Word2Vec(res, size=dim_vec,window=5,negative=10, alpha=0.01,iter=9,
                 min_count=5, workers=threads,sg=1,compute_loss=True)


# In[15]:


from gensim.test.utils import common_texts, get_tmpfile
path = get_tmpfile(path_w2v)
model.save(path_w2v)


# ## Glove 
# 

# In[48]:


if(not os.path.exists(glove_path)):
    os.makedirs(glove_path)
    subprocess.check_output("git clone https://github.com/stanfordnlp/GloVe",cwd=utilities_path,shell=True)
    subprocess.check_output("make",cwd =glove_path,shell=True)


# In[49]:


import subprocess
glove_corpus_path = glove_path + "press_all_glove.txt"
glove_script_file = "demo.sh"
glove_script_path = glove_path + glove_script_file
glove_vectors = glove_path + "vectors.txt"


# In[50]:


print('Start Trainin GloVe')
# Copy file into glove directory
subprocess.check_output(["cp " + train_path + " " + glove_corpus_path],shell=True)
# Copy the correct file to start gloVe with the right parameters
subprocess.check_output(["cp " +  utility_glove_script + " " + glove_script_path],shell=True)
# Start Training
subprocess.check_output(["sh", glove_script_file],cwd=glove_path)
# Transform File into w2v format (simplier to use during evlauation)
number_of_lines = subprocess.check_output("wc -l " + glove_vectors , shell=True).split()[0].decode('utf-8')
subprocess.check_output("sed -i '1i " + number_of_lines + " " + str(dim_vec) + "' " + glove_vectors,shell=True)
# Copy file into local models
subprocess.run(["cp", glove_vectors, model_path_glove])

