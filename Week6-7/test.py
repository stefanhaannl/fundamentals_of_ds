import numpy
import pandas as pd
import matplotlib.pyplot as plt
import pprint as pp
from pandas.tools.plotting import scatter_matrix

def init(path):
    image_df = pd.read_pickle(path)
    user_id =  image_df['user_id'].astype(int)
    df_init = pd.DataFrame({'image_id': image_df['image_id'], 'user_id':user_id })
    return df_init


def survey_column(column_type, init_df, data_df):
    """
    Returns column list of data_df specified with column_type
    """
    column_list = []
    i=0
    for i in range(0, len(init_df)):
        print (float(i)/float(len(init_df)))
        userid = init_df.user_id.iloc[i]
        column = data_df.loc[data_df['string_users'] == userid]
        column_value2 = column[column_type].tolist()[0]
        column_list.append(column_value2)
    
    return column_list
    

       
    
def survey(path, init_df):
    """
    
    """
    df = pd.read_pickle(path)
    pp.pprint(df)
    user_ids = df.insta_user_id.astype(int)
    df['string_users'] = user_ids
    
    # get values
    plist = survey_column('P', init_df, df)
    elist = survey_column('E', init_df, df)
    rlist = survey_column('R', init_df, df)
    mlist = survey_column('M', init_df, df)
    alist = survey_column('A', init_df, df)
    permalist = survey_column('PERMA', init_df, df)
    
    # add columns
    init_df['P'] = plist
    init_df['E'] = elist
    init_df['R'] = rlist
    init_df['M'] = mlist
    init_df['A'] = alist
    init_df['PERMA'] = permalist
    
    return init_df
    
    
# In[main]
if __name__ == "__main__": 
    
    paths = ['data/anp.pkl','data/face.pkl','data/image_data.pkl',
         'data/image_metrics.pkl','data/object_labels.pkl',
         'data/survey.pkl','data/object_labels.pkl']

    init_df = init(paths[2])
    image_df = image_metrics(paths[3], init_df)
    #survey_df = survey(paths[5], init_df)
    #survey_df.to_pickle('perma_features.pkl')
    
    

    
