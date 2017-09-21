import numpy as np
import pandas as pd
import string
import pickle
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer

def clean(doc):
    """
    Remove stopwords from string and normalize words in string
    INPUT: List of strings(which are the documents)
    OUTPUT: List of list of words
    """
    # Create a set of stopwords
    stop = set(stopwords.words('english'))

    # This is the function makeing the lemmatization
    lemma = WordNetLemmatizer()
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
    pass

<<<<<<< HEAD
def testDoc():
    doc1 = "Working out is great for the body. Fitness makes you feel good."
    doc2 = "Red cars are faster than blue cars."
    doc3 = "Doctors suggest that fitness increases muscle mass and speeds up metabolism."
    doc4 = "Cars with electrical engines cause less polution than cars with internal combustion engines."
    doc5 = "Pushups make a good upper body excercise."

    # compile documents
    doc_complete = [doc1, doc2, doc3, doc4, doc5]
    doc_clean = [clean(doc).split() for doc in doc_complete]
    return doc_clean
=======
# Create a set of stopwords
stop = set(stopwords.words('english'))

# This is the function makeing the lemmatization
lemma = WordNetLemmatizer()

doc1 = "Working out is great for the body. Fitness makes you feel good."
doc2 = "Red cars are faster than blue cars."
doc3 = "Doctors suggest that fitness increases muscle mass and speeds up metabolism."
doc4 = "Cars with electrical engines cause less polution than cars with internal combustion engines."
doc5 = "Pushups make a good upper body excercise."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]
doc_clean = [clean(doc).split() for doc in doc_complete] 
>>>>>>> b9da38578cdd486a015e5c47c277a9f3ffff7444

