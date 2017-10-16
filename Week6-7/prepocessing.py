"""
Daniel is een baas, Amor is een bitch
"""

import numpy
import pandas as pd
import matplotlib.pyplot as plt
import pprint as pp
from pandas.tools.plotting import scatter_matrix

# In[]
def init(path):
    image_df = pd.read_pickle(path)
    user_id =  image_df['user_id'].astype(str)
    df_init = pd.DataFrame({'image_id': image_df['image_id'], 'user_id':user_id })
    return df_init

# In[anp]

def anp(path, init_df):
    anp = pd.read_pickle(path)

    return anp_df
    
# In[face]

def face(path, init_df):

    
# In[image_data]

def image_data(path, init_df):

    
# In[image_metrics]

def image_metrics(path, init_df): 
    
    
# In[survey]

def survey(path, init_df):
    
    
# In[object_labels]

def object_labels(path, init_df):

    
# In[main]
if __name__ == "__main__": 
    
    paths = ['data/anp.pkl','data/face.pkl','data/image_data.pkl',
         'data/image_metrics.pkl','data/object_labels.pkl',
         'data/survey.pkl','data/object_labels.pkl']

    init_df = init(paths[2])
    
    # Load al the data
    anp_df = anp(paths[0])
    face_df = anp(paths[1])
    image_data_df = anp(paths[2])
    image_metrics_df = anp(paths[3])
    survey_df = anp(paths[4])
    object_labels_df = anp(paths[5])
    
    

    
