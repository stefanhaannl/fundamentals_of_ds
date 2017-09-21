import numpy as np
import pandas as pd

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
