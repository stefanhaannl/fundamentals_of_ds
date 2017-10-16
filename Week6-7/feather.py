# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:43:46 2017

@author: Daniel, Stefan
"""

import pandas as pd

# Load data
#def load_data(paths):
#    for i in range(0,len(paths)):
#        path = paths[i]
#        name = str(path)[5:-4]
#        return pd.read_pickle(path)

def load_data(path):
    
#    path = 'data/image_data.pkl'
    return pd.read_pickle(path)

    #return  df#1,df2,df3,df4,df5,df6
# In[]

paths = ['data/anp.pkl','data/face.pkl','data/image_data.pkl',
         'data/image_metrics.pkl','data/object_labels.pkl',
         'data/survey.pkl','data/object_labels.pkl']
#dfnames = [df1,df2,df3,df4,df5,df6]
# Load al the data
anp = load_data(paths[0])
face = load_data(paths[1])
image_data = load_data(paths[2])
image_metrics = load_data(paths[3])
survey = load_data(paths[4])
object_labels = load_data(paths[5])

# feather.write_dataframe(df, path)

# Dit zijn de user instagram id's = object_labels.insta_user_id
# volgens mij klopt het niet in deze format dus moet je nog int() doen.


# Voorbeeld
#im_id = image_data.image_id.iloc[9208]
im_id = '951728575726873168_289794729'
print image_data.loc[image_data.image_id == im_id]
print anp.loc[anp.image_id == im_id]
print face.loc[face.image_id == im_id]
print image_metrics.loc[image_metrics.image_id == im_id]
print survey.loc[survey.image_id == im_id]

# https://www.instagram.com/p/BJGysPxgsTy/
#print anp.loc[anp.image_id == image_data.iloc[0,0]]
#print face.loc[face.image_id == image_data.iloc[0,0]]
#print image_metrics.loc[image_metrics.image_id == image_data.iloc[0,0]]
#print survey.loc[survey.image_id == image_data.iloc[0,0]]