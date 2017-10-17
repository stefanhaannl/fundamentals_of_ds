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

def face_features(path, init_df):
    facepath = 'data/face_features.pkl'
    face = pd.read_pickle(facepath)

    return face
    
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
    likespath = 'data/likes_comments_features.pkl'
    image_data_df = pd.read_pickle(likespath)
    return image_data_df
    
# In[image_metrics]
def image_data_Axel(path, df):
    ### adds a lot of metadata of the profiles and photos
    ### INPUT: the current DF with all pictures as a row, the path to the df with meta per photo 
    ### OUTPUT: the input dataframe, but with information of the metadata
    image_data_df = pd.read_pickle(path)
    df["image_height"] = image_data_df["image_height"]
    df["image_link"] = image_data_df["image_link"]
    df["image_width"] = image_data_df["image_width"]
    df["data_memorability"] = image_data_df["data_memorability"]
    df["user_followed_by"] = image_data_df["user_followed_by"]
    df["user_follows"] = image_data_df["user_follows"]
    df["user_posted_photos"] = image_data_df["user_posted_photos"]
    indexes = image_data_df.index[(image_data_df["image_filter"] != "Normal")].tolist()
    subset = image_data_df.loc[indexes]["image_id"].tolist()
    newFeature = [0] * df.shape[0]
    #print(df.index[df['image_id'].isin(subset)].tolist())
    for image in df.index[df['image_id'].isin(subset)].tolist():
        newFeature[image ] +=1
    df["special_filter"] = newFeature
    return df
    
# In[survey]

def survey(path, init_df):
    perma_path = 'data/perma_features.pkl'
    survey_df = pd.read_pickle(perma_path)
    return survey_df
    
# In[object_labels]

def object_labels(path, df, threshold = 0.0, amountFeaturesToAdd = 20):
    ### adds an amount of boolean features which indicate whether an object is in the photo
    ### INPUT: the current DF with all pictures as a row, the path to the DF with the object per photo, potentially 
    ### a threshold for when to include objects, and the amount of boolean features added
    ### OUTPUT: the input dataframe, but with information of which object are in the photo
    object_labels_df = pd.read_pickle(path)
    features = object_labels_df["data_amz_label"].value_counts().keys()[:amountFeaturesToAdd]
    for feature in features:
        indexes = object_labels_df.index[(object_labels_df["data_amz_label"] == feature) & 
                                         (object_labels_df["data_amz_label_confidence"] > threshold)].tolist()
        subset = object_labels_df.loc[indexes]["image_id"].tolist()
        newFeature = [0] * df.shape[0]
        for user in df.index[df['image_id'].isin(subset)].tolist():
            newFeature[user] +=1
        df[feature] = newFeature    
    return df
    
# In[main]
if __name__ == "__main__": 
    
    paths = ['data/anp.pkl','data/face.pkl','data/image_data.pkl',
         'data/image_metrics.pkl','data/survey.pkl','data/object_labels.pkl']

    init_df = init(paths[2])
    
    # Load al the data
    #anp_df = anp(paths[0], init_df)
    face_df = face_features(paths[1], init_df)
    image_data_df = image_data(paths[2], init_df)
    image_metrics_df = image_data_Axel(paths[2], init_df)
    survey_df = survey(paths[4], init_df)
    object_labels_df = object_labels(paths[5], init_df)
    

    

    
