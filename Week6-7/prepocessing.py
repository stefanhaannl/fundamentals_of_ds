
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import pprint as pp
from pandas.tools.plotting import scatter_matrix

def init(path):
    image_df = pd.read_pickle(path)
    df = pd.DataFrame({'image_id': image_df['image_id']})
    return df

if __name__ == "__main__": 
    path = 'pickles/image_data.pkl'
    df = init(path)
    pp.pprint(df)
    
