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
    
    return face_df
    
# In[image_data]

def image_data(path, init_df):
    
    return image_data_df
    
# In[image_metrics]

def image_metrics(path, init_df): 
    
    return image_metrics_df
    
# In[survey]

def survey(path, init_df):
    perma_path = 'data/perma_features.pkl'
    survey_df = pd.read_pickle(perma_path)
    return survey_df
    
# In[object_labels]

def object_labels(path, init_df):

    return object_labels_df
    
# In[main]
if __name__ == "__main__": 
    
    paths = ['data/anp.pkl','data/face.pkl','data/image_data.pkl',
         'data/image_metrics.pkl','data/survey.pkl','data/object_labels.pkl']

    init_df = init(paths[2])
    
    # Load al the data
    #anp_df = anp(paths[0], init_df)
    #face_df = face(paths[1], init_df)
    #image_data_df = image_data(paths[2], init_df)
    #image_metrics_df = image_metrics(paths[3], init_df)
    survey_df = survey(paths[4], init_df)
    print(type(survey_df.P.iloc[0]))
    print(type(survey_df.E.iloc[0]))
    print(type(survey_df.R.iloc[0]))
    print(type(survey_df.M.iloc[0]))
    print(type(survey_df.A.iloc[0]))
    print(type(survey_df.PERMA.iloc[0]))
    
    #object_labels_df = object_labels(paths[5], init_df)
    
    

    
