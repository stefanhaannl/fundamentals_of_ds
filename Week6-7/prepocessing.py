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
    face = pd.read_pickle(path)
    print "Merging the different faces emotions into one face per person per picture..."
    facegrouped = face.groupby(['image_id','face_id']).apply(merge_to_face)
    print "Dropping irrelevant fields..."
    del facegrouped['emo_confidence']
    del facegrouped['eyeglasses_confidence']
    del facegrouped['face_beard_confidence']
    del facegrouped['face_gender_confidence']
    del facegrouped['face_mustache_confidence']
    del facegrouped['face_smile_confidence']
    del facegrouped['face_id']
    del facegrouped['face_emo']
    del facegrouped['image_id']
    print "Merging the different faces to facedata per image..."
    image_facedata = facegrouped.groupby(level = 0).apply(merge_to_image)
    new_cols = ['ANGRY','CALM','CONFUSED','DISGUSTED','HAPPY','SAD','SURPRISED','age','beards','eyeglasses','male_percentage','mustaches','n_people','smiles','sunglasses']
    init_df.set_index('image_id',inplace=True)
    for col in new_cols:
        init_df[col] = numpy.nan
    print "Editing the image data and returning the initial df with added features..."
    for image in list(image_facedata.index.levels[0]):
        init_df.loc[image,new_cols] = image_facedata[image]     
    return init_df

def merge_to_image(x):
    ret = {}
    ret['n_people'] = len(x['face_gender'])
    if 'Male' in list(x['face_gender'].value_counts().index):
        ret['male_percentage'] = x['face_gender'].value_counts().loc['Male']/len(x['face_gender'])
    ret['age'] = (x['face_age_range_low']+(x['face_age_range_high']-x['face_age_range_low'])/2).mean()
    ret['sunglasses'] = x['face_sunglasses'].sum()
    ret['beards'] = x['face_beard'].sum()
    ret['mustaches'] = x['face_mustache'].sum()
    ret['smiles'] = x['face_smile'].sum()
    ret['eyeglasses'] = x['eyeglasses'].sum()
    emotions = ['HAPPY','SAD','SURPRISED','CONFUSED','ANGRY','CALM','DISGUSTED']
    for emotion in emotions:
        ret[emotion] = x[emotion].sum()/len(x['face_gender'])
    return pd.Series(ret)

def merge_to_face(x):
    ret = dict(x.iloc[0])
    emotions = ['HAPPY','SAD','SURPRISED','CONFUSED','ANGRY','CALM','DISGUSTED']
    face_emotions = list(x['face_emo'])
    for emotion in emotions:
        if emotion in face_emotions:
            ret[emotion] = 1
        else:
            ret[emotion] = 0
    return pd.Series(ret)  

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
    face_df = face(paths[1], init_df)
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
    
    

    
