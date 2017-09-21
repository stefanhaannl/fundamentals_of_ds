import numpy as np
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

def clean(doc):
    """
    Remove stopwords from string and normalize words in string
    INPUT: List of strings(which are the documents)
    OUTPUT: List of list of words
    """
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
    return normalized

def load_tweets(filepath):
    """
    Load an array of tweets two work with. (pickle file)
    INPUT: filepath to your pickle file
    OUTPUT: Pandas series of tweet documents (as lists)
    """
    df = pd.read_pickle(filepath)
    return df[['words','hashtags']]

def cluster_by_hashtags(df,n):
    """
    Generates a corpus based on the n'th most prevalent hashtags
    INPUT: Dataframe with hashtags (list) and text (list) column
    OUTPUT: Dataframe with hashtag (str) and text(list) column
    """
    result_df = []
    hashtags = []
    for lst in df['hashtags']:
        hashtags.extend(lst)
    hashtags = pd.Series(hashtags)
    hashtag_values = hashtags.value_counts()[0:n]
    hashtags_list = hashtag_values.index.values.tolist()
    for hashtag in hashtags_list:
        print "Filtering words for hashtag: "+hashtag
        words = []
        for row in df.itertuples():
            row = tuple(row)
            if hashtag in row[2]:
                words.extend(row[1])
        result_df.append([hashtag, words])
    return pd.DataFrame(result_df,columns=['hashtag','words'])
    



