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




