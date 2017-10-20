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

    userimage = []

    for i in anp.image_id:
        userimage.append(str(i).split('_')[1])
        
    anp['user_id'] = userimage
    
    # Create new dataframe, fill all emotions with NaN, then retrieve average
    # emotion score per emotion per user.
    uniq_users = init_df.user_id.drop_duplicates()
    uniq_users = uniq_users.reset_index().user_id
    users_df = pd.DataFrame()
    users_df['user_id'] = uniq_users
    users_df['anp_sen'] = np.nan
    users_df['acceptance'] = np.nan
    users_df['admiration'] = np.nan
    users_df['amazement'] = np.nan
    users_df['anger'] = np.nan
    users_df['annoyance'] = np.nan
    users_df['anticipation'] = np.nan
    users_df['apprehension'] =np.nan
    users_df['boredom'] = np.nan
    users_df['disgust'] = np.nan
    users_df['distraction'] = np.nan
    users_df['ecstasy'] = np.nan
    users_df['fear'] = np.nan
    users_df['grief'] = np.nan
    users_df['interest'] = np.nan
    users_df['joy'] = np.nan
    users_df['loathing'] = np.nan
    users_df['pensiveness'] = np.nan
    users_df['rage'] = np.nan
    users_df['sadness'] = np.nan
    users_df['serenity'] = np.nan
    users_df['surprise'] = np.nan
    users_df['terror'] = np.nan
    users_df['trust'] = np.nan
    users_df['vigilance'] = np.nan
        
        
    # Nu per user emotionscore bepalen
    for i in range(0,len(users_df)):
                print float(i)/float(len(users_df))
                userid = users_df.user_id[i]
        #        anp.emotion_score.groupby(['emotion_label']).mean()
                emo_score = anp[anp.user_id == userid].groupby(['emotion_label']).emotion_score.mean()
                for a in range(0,len(emo_score)):
                    users_df.iloc[i,users_df.columns == emo_score.index[a]] = emo_score[a]
                
                users_df.anp_sen[i] = anp[anp.user_id == userid].anp_sentiment.mean()
    
    anp_users = users_df
    anp_users.to_pickle('anp_users.pkl')   
    
# In[face]

def face_features(path, init_df):
    facepath = 'data/face_features.pkl'
    face = pd.read_pickle(facepath)
    face.reset_index(inplace=True)  

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
    for image in list(image_facedata.index):
        init_df.loc[image,new_cols] = image_facedata.loc[image]     
    return init_df

def merge_to_image(x):
    ret = {}
    ret['n_people'] = len(x['face_gender'])
    a=0
    for gender in list(x['face_gender']):
        if gender == 'Male':
            a+=1
    ret['male_percentage'] = float(a)/len(x['face_gender'])
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
    
def merge_data(face_df, image_data_df, image_metrics_df, survey_df, object_labels_df):
    # merge all the data 
    frame1 = pd.merge(image_metrics_df, image_data_df, how='inner', on='image_id')
    frame2 = pd.merge(frame1, survey_df, how='inner', on='image_id')
    frame3 = pd.merge(frame2, object_labels_df, how='inner', on='image_id')
    frame4  = pd.merge(frame3, face_df, how='inner', on='image_id')
    del frame4['user_id_x']
    frame4['user_id'] = frame4['user_id_y']
    del frame4['user_id_y']
    return frame4
    
def image_pickle(path):
    df = pd.read_pickle(path)
    return df


# In[Images to users]

def merge_user(image_df):
    userdf = image_df.groupby('user_id').apply(merge_user_apply)

    #filtering here
    return userdf

def merge_user_apply(x):
    ret = {}
    
    ret['image_height'] = x['image_height'].mean()
    ret['image_width'] = x['image_width'].mean()
    ret['data_memorability'] = x['data_memorability'].mean()
    ret['user_followed_by'] = max(x['user_followed_by'])
    ret['user_follows'] = max(x['user_follows'])
    ret['user_posted_photos'] = max(x['user_posted_photos'])
    ret['filter'] = x['special_filter'].mean()
    ret['comments'] = x['comment_count'].mean()
    ret['likes'] = x['like_count'].mean()
    lmeans = ['P','E','R','M','A','PERMA','Person','People','Human','Poster','Plant','Portrait','Flyer','Face','Animal','Food','Smile','Brochure','Mammal','Text','Potted Plant','Pet','Furniture','Collage','Outdoors','Canine','ANGRY','CALM','CONFUSED','DISGUSTED','HAPPY','SAD','SURPRISED','beards','eyeglasses','male_percentage','mustaches']
    for lmean in lmeans:
        ret[lmean] = x[lmean].mean()
    ret['age_mean'] = x['age'].mean()
    ret['age_max'] = numpy.nanmax(x['age'])
    ret['age_min'] = numpy.nanmin(x['age'])
    ret['people_mean'] = x['n_people'].mean()
    ret['people_max'] = numpy.nanmax(x['n_people'])
    ret['smiles'] = (x['smiles']/x['n_people']).mean()
    ret['sunglasses'] = (x['sunglasses']/x['n_people']).mean()
    return pd.Series(ret)


# In[Filter final dataframe]

def filter_user_features(df):
    #user posted photos >= 5
    df = df[df['user_posted_photos'] >= 5]
    #change the min en max age values
    df['age_min'] = df['age_min'].apply(lambda x: x.iloc[0])
    df['age_max'] = df['age_max'].apply(lambda x: x.iloc[0])
    df['people_max'] = df['people_max'].apply(lambda x: x.iloc[0])
    #rename some columns
    df.rename(columns={'likes':'image_likes','comments':'image_comments','filter':'image_filter','data_memorability':'image_data_memorability','Animal':'object_animal','Brochure':'object_borchure','Canine':'object_canine','Collage':'object_collage','Face':'object_face','Flyer':'object_flyer','Food':'object_food','Furniture':'object_furniture','Human':'object_human','Mammal':'object_mammal','Outdoors':'object_outdoors','People':'object_people','Person':'object_person','Pet':'object_pet','Plant':'object_plant','Portrait':'object_portrait','Poster':'object_poster','Potted Plant':'object_potted_plant','Text':'object_text','Smile':'object_smile','ANGRY':'face_emotion_angry','CALM':'face_emotion_calm','CONFUSED':'face_emotion_confused','DISGUSTED':'face_emotion_disgusted','HAPPY':'face_emotion_happy','SAD':'face_emotion_sad','SURPRISED':'face_emotion_surprised','age_max':'face_age_max','age_mean':'face_age_mean','age_min':'face_age_min','beards':'face_beards','eyeglasses':'face_eyeglasses','male_percentage':'face_male_percentage','mustaches':'face_mustaches','people_max':'face_people_max','people_mean':'face_people_mean','smiles':'face_smiles','sunglasses':'face_sunglasses','P':'outcome_P','E':'outcome_E','R':'outcome_R','M':'outcome_M','A':'outcome_A','PERMA':'outcome_PERMA'},inplace=True)
    #nan value desicions:
    df['face_people_max'].fillna(0,inplace=True)
    df['face_people_mean'].fillna(0,inplace=True)
    for col in list(df):
        if col[0:3] == 'anp':
            df[col].fillna(0,inplace=True)
        if col[0:4] == 'face':
            df[col].fillna(df[col].mean(),inplace=True)
    
    return df


def new_perma(dfn,survey):
    survey = survey[['P_1','P_2','P_3','E_1','E_2','E_3','R_1','R_2','R_3','M_1','M_2','M_3','A_1','A_2','A_3','insta_user_id']]
    survey['P'] = (survey['P_1'].apply(float)+survey['P_2']+survey['P_3'])/3
    survey['E'] = (survey['E_1'].apply(float)+survey['E_2']+survey['E_3'])/3
    survey['R'] = (survey['R_1'].apply(float)+survey['R_2']+survey['R_3'])/3
    survey['M'] = (survey['M_1'].apply(float)+survey['M_2']+survey['M_3'])/3
    survey['A'] = (survey['A_1'].apply(float)+survey['A_2']+survey['A_3'])/3
    survey['PERMA'] = (survey['P']+survey['E']+survey['R']+survey['M']+survey['A'])/5
    survey = survey[['P','E','R','M','A','PERMA','insta_user_id']]
    
    survey['insta_user_id'] = survey['insta_user_id'].apply(int)

    dfn.drop(['outcome_P','outcome_E','outcome_R','outcome_M','outcome_A','outcome_PERMA'],axis=1,inplace=True)
    
    survey = survey[survey['id'].isin(list(dfn['user_id']))]
    survey = survey[~survey['insta_user_id'].duplicated()]
    survey.rename(columns={'insta_user_id':'user_id'},inplace=True)
    
    newdfn = pd.merge(dfn, survey, how='inner', on='user_id')
    newdfn.rename(columns={'P':'outcome_P','E':'outcome_E','R':'outcome_R','M':'outcome_M','A':'outcome_A','PERMA':'outcome_PERMA',},inplace=True)
    
    return newdfn
# In[main]
if __name__ == "__main__": 
    


    #read the user features data
    #df = pd.read_pickle('data/user_features.pkl')
    #df.reset_index(inplace=True)
    #read the anp data
    #anp = pd.read_pickle('data/anp_users.pkl')
    #anp['user_id'] = anp['user_id'].apply(int)
    #change the column names
    #anp.rename(columns={'anp_sen':'sen'},inplace=True)
    #cols = list(anp)
    #cols.remove('user_id')
    #for column in cols:
    #    anp.rename(columns={column:'anp_'+column},inplace=True)
    #merge the user features with the anp features
    #dfn = pd.merge(df, anp,how='inner', on='user_id')
    #filter some users based on features and fill nan values
    #dfn = filter_user_features(dfn)
    dfn = pd.read_pickle('data/final_user_features.pkl')
    
    
    