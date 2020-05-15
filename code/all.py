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




#!/usr/bin/env python
# coding: utf-8

# # Train

# ## SetUp directories

# In[1]:


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
model = Word2Vec(res, size=300,window=5,negative=10, alpha=0.01,iter=9,
                 min_count=5, workers=4,sg=1,compute_loss=True)


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

#!/usr/bin/env python
# coding: utf-8

# ## SetUp Parameters

# In[129]:


utilities_path = '../utilities/'
model_path = '../models/'
data_path = '../data/'
results_path = '../results/'


# In[134]:


from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y")
dt_string = "13-05-2020"
global_path = "../models/"
model_path = ".model"
txt_extension = ".txt"
path_ft =  global_path + "ft_"+ dt_string + txt_extension
path_w2v =  global_path + "w2v_"+ dt_string + model_path
path_glove = global_path + "gloVe_"+ dt_string + txt_extension


# ### Import Models

# In[135]:


from gensim.test.utils import common_texts, get_tmpfile,datapath
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
model_ft = KeyedVectors.load_word2vec_format(path_ft)


# In[136]:


from gensim.test.utils import common_texts, get_tmpfile,datapath
from gensim.models import Word2Vec
path = get_tmpfile(path_w2v)
model_w2v = Word2Vec.load(path_w2v)
model_w2v = model_w2v.wv


# In[137]:


from gensim.test.utils import common_texts, get_tmpfile,datapath
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
model_glove = KeyedVectors.load_word2vec_format(path_glove)


# In[138]:


model_glove.vocab


# ### Import DataSet

# In[139]:


import pandas as pd 
df = pd.read_csv('../data/eval.csv')


# In[140]:


#df = (df[(df['legal_name'] == "THERADIAG SA")])
#df=df.sample(10)


# In[141]:


df


# In[142]:


# Add columns to store results

df['eval_set_w2v'] = ""
df['eval_set_ft'] = ""
df['eval_set_glove'] = ""
df['eval_number_w2v'] = ""
df['eval_number_ft'] = ""
df['eval_number_glove'] = ""
df['corpus'] = df['corpus'].apply(lambda corpus: set(corpus.split(" ")))


# In[143]:


# Get name of all companies in eval dataset 
vocab = set(df.legal_name)


# In[144]:


# Map SIREN to legal name and vice versa
from collections import defaultdict
siren_to_legal_name = defaultdict(str)
legal_n_to_siren = defaultdict(str)
for i,row in df.iterrows():
    siren_to_legal_name[row.siren] = row.legal_name
    legal_n_to_siren[row.legal_name] = row.siren


# ### Create Result Dictionnary

# In[145]:


import pickle
# Load common names dictionnary
file = open('../utilities/legal_to_common_names.obj', 'rb') 
legal_to_common_name = pickle.load(file)


# In[146]:


def add_eval_set(dataframe, legal_name, nearest_words,method):
    occurences_word = defaultdict(int)
    df_legal_name = dataframe[(dataframe['legal_name'] == legal_name)]
    for i,row in df_legal_name.iterrows():
        nearest_words_set = set(nearest_words)
        corpus_set = row.corpus
        inters_ = nearest_words_set.intersection(corpus_set)
        if(method == 'w2v'):
            df['eval_set_w2v'][i] =  list(inters_)
            df['eval_number_w2v'][i] = len(inters_)
        elif(method == 'ft'):
            df['eval_set_ft'][i] =  list(inters_)
            df['eval_number_ft'][i] = len(inters_)
        else:
            df['eval_set_glove'][i] =  list(inters_)
            df['eval_number_glove'][i] = len(inters_)
        for word in inters_:
                occurences_word[word] +=1
    return occurences_word


# In[147]:


def create_emb_dict(word):
    emb_dict = defaultdict(list)
    emb_dict['ft'] = get_most_similar(word,model_ft)
    emb_dict['w2v'] = get_most_similar(word,model_w2v)
    emb_dict['glove'] = get_most_similar(word,model_glove)
    return emb_dict


# In[148]:


def get_nearest_ft(word):
    nearest_words  = model_ft.get_nearest_neighbors(word.lower(),10)
    return [[(y),round(x,2)] for x,y in nearest_words]


# In[149]:


import numpy as np
def get_composed_word_vector(composed_word,wordVectors): 
    res = []
    for word in composed_word.split():
        if(word.lower() in wordVectors.vocab):
            res.append(wordVectors[word.lower()])
    if(len(res)==0):
        return res 
    return np.mean(res,axis=0)


# In[150]:


def get_most_similar(word,model,topn=10):
    res = []
    if(len(word.split(" ")) > 0):
        vector = get_composed_word_vector(word,model)
    else: 
        vector = model[word.lower()]
    if(len(vector)>0):
        nearest_words = model.most_similar([vector], topn=topn)
        res =  [[(x),round(y,2)] for x,y in nearest_words]
    return res


# In[151]:


def score2(legal_name,word_occurences):
    return round((word_occurences / len(df[(df['legal_name'] == legal_name)])),4)
               


# In[152]:


def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices


# In[153]:


def compute_score2(dataframe,nearest_words, legal_name,method):
    occurences_words = add_eval_set(df, legal_name,[x[0] for x in nearest_words],method)
    for word in nearest_words: 
        word.append(score2(legal_name, occurences_words[word[0]]))


# In[154]:


def get_nearest_from_dict(dict_,legal_name,method):
    res = []
    common_names_dict = dict_[legal_name] 
    for common_name in common_names_dict.keys():
        #print(dict_[legal_name][common_name])
        res.append(dict_[legal_name][common_name][method])
    return [y for x in res for y in x]


# In[155]:


def getTime(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    time_since_start = "Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    return time_since_start


# In[156]:


def fusion(nearest_words):
    word_list = [x[0] for x in nearest_words]
    res = []
    list_of_pairs = []
    for i,word in enumerate(word_list):
        indices_word = all_indices(word,word_list)
        list_of_pairs.append(indices_word)
    list_of_pairs = [list(item) for item in set(tuple(row) for row in list_of_pairs)]
    for pairs in list_of_pairs:
        if(len(pairs)>1):
            res.append(fusion_list([x for j,x in enumerate(nearest_words) if j in pairs ]))
        else: 
            res.append(nearest_words[pairs[0]])
    return res
    


# In[157]:


def fusion_list(l):
    new_similarity = round(sum([x[1] for x in l]),3)
    return [l[0][0], new_similarity, l[0][2]]


# In[158]:


import time
nearest = defaultdict(dict)
new_nearest = defaultdict(dict)
methods = ['ft','w2v','glove']
# Create dictionnary with nearest vectors
percent = 0
start = time.time()
for i,legal_name in enumerate(vocab): 
    # Logging
    if(i % 150 == 0):
        time_ = getTime(start,time.time())
        print("0%" + "=" *(int(percent/10))+ str(percent) +"%, " + time_, end="\r" )
        percent +=10
    legal_name_dict = defaultdict(dict)
    nearest[legal_name] = legal_name_dict
    siren = legal_n_to_siren[legal_name]
    if(siren in legal_to_common_name.keys()):
        common_names = legal_to_common_name[siren]
        # For each common name compute nearest words per embedding technique 
        for common_n in common_names:
            legal_name_dict[common_n] = create_emb_dict(common_n)     
    else:
        legal_name_dict[legal_name] = emb_dict
    legal_name_dict = defaultdict(list)
    new_nearest[legal_name] = legal_name_dict
    for method in methods:
        fusioned_dict = defaultdict(list)
        # Compute score2
        compute_score2(df,get_nearest_from_dict(nearest,legal_name,method),legal_name,method)
        # Fusion result of each common name
        nearest_fusioned = fusion(get_nearest_from_dict(nearest,legal_name,method))
        #nearest_fusioned
        legal_name_dict[method] = nearest_fusioned
        # Compute Gobal Score to evaluate method
        sgx1 = round(sum(x[1] for x in nearest_fusioned),3)
        sgx2 = round(sum(x[2]for x in nearest_fusioned),3)
        sgx3 = sgx1*sgx2
        scores = [('sgx1',sgx1),('sgx2',sgx2),('sgx3',sgx3)]
        legal_name_dict[method].insert(0,scores)


# In[159]:


new_nearest


# In[160]:


df.sample(20)


# In[161]:


df.to_csv('results' + dt_string + '.csv')


# In[162]:


ftEval = df['eval_number_ft'].mean()
w2vEval = df['eval_number_w2v'].mean()
gloveEval = df['eval_number_glove'].mean()
print(ftEval)
print(w2vEval)
print(gloveEval)


# In[37]:


import json
import codecs

with codecs.open('results-with-fusion.json', 'w',encoding='utf-8') as fp:
    json.dump(nearest,fp,ensure_ascii=False)


# ### Prediction 

# In[41]:


# DataSetPath 
prediction_path = utilities_path + ''


# In[101]:


df_prediction = pd.read_csv('../data/prediction.csv')
df_prediction
df_prediction['legal_name'] = df_prediction['siren'].apply(lambda siren: siren_to_legal_name[siren])
df_prediction['prediction'] = ""
df_prediction


# In[78]:


vocab_predictions  = set(df_prediction.siren)
                         


# In[120]:


siren_to_emb = defaultdict()
for siren in vocab_predictions: 
    legal_name = siren_to_legal_name[siren]
    vec_legal_name = get_composed_word_vector(legal_name,model_glove)  
    siren_to_emb[siren] = vec_legal_name


# In[126]:


from operator import itemgetter
def most_similar_legal_name(corpus,model):
    vector_corpus = get_composed_word_vector(corpus,model)
    results = []
    for siren in vocab_predictions:
        vec_legal_name = siren_to_emb[siren]
        if(len(vec_legal_name)>0):
            result = spatial.distance.cosine(vector_corpus, vec_legal_name)
            results.append((result,siren_to_legal_name[siren]))
    final_res = min(results,key=itemgetter(0)) 
    return final_res[1]


# In[127]:


for i,row in df_prediction.iterrows():
    if(i%250 == 0):
        print("10%done")
    df_prediction.prediction[i] = most_similar_legal_name(row['corpus'],model_glove)


# In[128]:


df_prediction


# In[ ]:




