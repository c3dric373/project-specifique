#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
filename = sys.argv[1]
print(filename + "ntm")

# In[ ]:


import fasttext as ft
model = ft.load_model(filename)


# In[120]:


import pandas as pd
df = pd.read_csv('../data/corpus_check.csv')


# In[121]:


df


# In[122]:


import re
import contractions
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import FrenchStemmer
import spacy
from spacy_lefff import LefffLemmatizer, POSTagger

nlp = spacy.load('fr')
nlp.max_length = 68111459
french_lemmatizer = LefffLemmatizer()
nlp.add_pipe(french_lemmatizer, name='lefff')
translator = str.maketrans('', '', string.punctuation)


# In[123]:


translator = str.maketrans('', '', string.punctuation)

def preprocessing(doc):
        result = []
        doc = doc.replace('\n', ' ')
        # To lowercase
        doc = doc.lower()
        # Remove numbers
        doc = re.sub("\d+", "", doc)
        doc = doc.replace('m€', '')
        doc = doc.replace('k€', '')
        sentences = sent_tokenize(doc)
        # Lemmentization (same as stemming but based on knowledge base)
        for sent in sentences:
            # Delete punctuation
            sent = sent.translate(translator)
            lemmentization = []
            doc_ = nlp(sent,disable = ['ner', 'parser'])
            for d in doc_:
                lemmentization.append(d.lemma_)
            lemmentization.append('\n')
            res_ = " ".join(lemmentization)
            result.append(res_)
        return result

def test(dataframe,index=0):
    f = open('' + str(index) + '.txt' , 'w')
    for i,row in dataframe.iterrows():
        result = "".join(preprocessing((row['text'])))
        f.write(result)  # python will convert \n to os.linesep
    f.close()


# In[124]:


#df=df.sample(10)



# In[127]:


# Get name of all companies in eval dataset
vocab = set(df.legal_name)


# In[128]:


from collections import defaultdict
nearest = defaultdict(str)
# Very important to use lower here, as we lower everything during preprocessing
for x in vocab:
    nearest[x.lower()] = model.get_nearest_neighbors(x.lower(),5)


# In[131]:


for i,row in df.iterrows():
    nearest_words = set([y[1:len(y)-2] for x,y in (nearest[row.legal_name.lower()])])
    inters_ = nearest_words.intersection(row.corpus)
    df['eval_set'][i] =  inters_
    df['eval_number'][i] = len(inters_)




# In[133]:




# In[ ]:




