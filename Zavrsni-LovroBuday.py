import urllib.request as req
import tarfile
import os

imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

save_filename = "aclImdb_v1.tar.gz"

if not os.path.exists(save_filename):
    req.urlretrieve(imdb_url, save_filename)

imdb_folder = "aclImdb"
if not os.path.exists(imdb_folder):
    with tarfile.open(save_filename) as tar:
        tar.extractall()
        

#%%
        
import re
import numpy as np

def get_reviews(data_folder="/train"):
    reviews = []
    labels = []
    for index, sentiment in enumerate([ "/neg/", "/pos/"]):
        path = imdb_folder + data_folder + sentiment
        for filename in sorted(os.listdir(path)):
            with open(path + filename, 'r', encoding="utf8") as f:
                review = f.read()
                review = review.lower()
                review = review.replace("<br />", " ")
                review = re.sub(r"[^a-z ]", " ", review)
                review = re.sub(r" +", " ", review)
                review = review.split(" ")
                reviews.append(review)
                
                label = [0, 0]
                label[index] = 1
                labels.append(label)
                
    return reviews, np.array(labels)
    
train_reviews, train_labels = get_reviews()
print(len(train_reviews))

#%%

import zipfile

glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"

save_filename = "glove.6B.zip"
if not os.path.exists(save_filename):
    req.urlretrieve(glove_url, save_filename)
    
EMBEDDING_SIZE = 50

glove_filename = "glove.6B.{}d.txt".format(EMBEDDING_SIZE)
if not os.path.exists(glove_filename) and EMBEDDING_SIZE in [50,100,200,300]:
    with zipfile.ZipFile(save_filename, 'r') as z:
        z.extractall()

#%%
from collections import defaultdict

def load_embeddings():
    with open(glove_filename, 'r', encoding="utf8") as glove_vectors:
        word_to_int = defaultdict(int)
        int_to_vec = defaultdict(lambda: np.zeros([EMBEDDING_SIZE]))
        
        index = 1
        for line in glove_vectors:
            fields = line.split()
            word = str(fields[0])
            vec = np.asarray(fields[1:], np.float32)
            word_to_int[word] = index
            int_to_vec[index] = vec
            index += 1
    return word_to_int, int_to_vec

word_to_int, int_to_vec = load_embeddings()

def review_words_to_ints(train_reviews):
    train_data = []
    for review in train_reviews:
        int_review = [word_to_int[word] for word in review]
        train_data.append(int_review)
    return train_data

train_reviews = review_words_to_ints(train_reviews)
print(train_reviews[0])


MAX_REVIEW_LEN = 500

def zero_pad_reviews(train_reviews):
    train_data_padded = []
    for review in train_reviews:
        padded = [0] * MAX_REVIEW_LEN
        stop_index = min(len(review), MAX_REVIEW_LEN)
        padded[:stop_index] = review[:stop_index]
        train_data_padded.append(padded)
    return train_data_padded

train_reviews = zero_pad_reviews(train_reviews)
print(train_reviews[0])

#%%


import pickle

with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([train_reviews, train_labels], f)
    
"""
with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
    obj0, obj1, obj2 = pickle.load(f)
"""

#%%

def review_ints_to_vecs(train_reviews):
    train_data = []
    for review in train_reviews:
        vec_review = [int_to_vec[word] for word in review]
        train_data.append(vec_review)
    return train_data

train_reviews = np.array(review_ints_to_vecs(train_reviews))
print(train_reviews.shape)


    