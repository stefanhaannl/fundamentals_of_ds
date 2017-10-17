"""
Daniel is een baas, Amor is een bitch
"""

import numpy as np
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

    init_df['anp_sen'] = np.nan
    init_df['acceptance'] = np.nan
    init_df['admiration'] = np.nan
    init_df['amazement'] = np.nan
    init_df['anger'] = np.nan
    init_df['annoyance'] = np.nan
    init_df['anticipation'] = np.nan
    init_df['apprehension'] =np.nan
    init_df['boredom'] = np.nan
    init_df['disgust'] = np.nan
    init_df['distraction'] = np.nan
    init_df['ecstasy'] = np.nan
    init_df['fear'] = np.nan
    init_df['grief'] = np.nan
    init_df['interest'] = np.nan
    init_df['joy'] = np.nan
    init_df['loathing'] = np.nan
    init_df['pensiveness'] = np.nan
    init_df['rage'] = np.nan
    init_df['sadness'] = np.nan
    init_df['serenity'] = np.nan
    init_df['surprise'] = np.nan
    init_df['terror'] = np.nan
    init_df['trust'] = np.nan
    init_df['vigilance'] = np.nan
    
    i = 0
    for i in range(0,len(init_df.image_id)):
        print float(i)/float(len(init_df.image_id))
#        anp.emotion_score.groupby(['emotion_label']).mean()
        emo_score = anp[anp.image_id == init_df.image_id.loc[i]].groupby(['emotion_label']).emotion_score.mean()
        for a in range(0,len(emo_score)):
            init_df.iloc[i,init_df.columns == emo_score.index[a]] = emo_score[a]
        init_df.anp_sen[i] = anp[anp.image_id == init_df.image_id.loc[i]].anp_sentiment.mean()
        
    return init_df
    
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
            newFeature[user-1] +=1
        df[feature] = newFeature    
    return df
    
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
    
    

    
