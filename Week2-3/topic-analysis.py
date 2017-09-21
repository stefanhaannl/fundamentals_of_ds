import numpy as np
import pandas as pd
<<<<<<< HEAD
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
    return normalized

doc1 = "Working out is great for the body. Fitness makes you feel good."
doc2 = "Red cars are faster than blue cars."
doc3 = "Doctors suggest that fitness increases muscle mass and speeds up metabolism."
doc4 = "Cars with electrical engines cause less polution than cars with internal combustion engines."
doc5 = "Pushups make a good upper body excercise."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]
doc_clean = [clean(doc).split() for doc in doc_complete] 
=======

def load_tweets(filepath):
    """
    Load an array of tweets two work with. (pickle file)
    INPUT: filepath to your pickle file
    OUTPUT: Pandas series of tweet documents (as lists)
    """
    df = pd.read_pickle(filepath)
    return df[['text','hashtags']]
>>>>>>> 4c70b451cf31cb574b0e782bde3159d04cb34a67
