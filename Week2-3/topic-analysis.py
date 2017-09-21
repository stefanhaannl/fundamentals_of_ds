import numpy as np
import pandas as pd

def load_tweets(filepath):
    """
    Load an array of tweets two work with. (pickle file)
    INPUT: filepath to your pickle file
    OUTPUT: Pandas series of tweet documents (as lists)
    """
    df = pd.read_pickle(filepath)
    return df[['text','hashtags']]