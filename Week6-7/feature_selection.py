import numpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('ggplot')
import pprint as pp
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
import random
import matplotlib.ticker as ticker

    
def correlation_matrix(df):
    """
    Input: dataframe
    Output: correlation matrix
    """
    return df.corr(method='pearson')

def plot_correlation(corr_matrix):
    """
    Input: correlation matrix
    Output: plots of the corr matrix
    """
    headers = list(df.columns.values)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_matrix)
    fig.colorbar(cax)
    plt.xticks(rotation=90)
    ax.set_xticklabels([''] + headers)
    ax.set_yticklabels([''] + headers)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    
def normalize_df(df):
    """
    Input: correlation matrix
    Output: plots of the corr matrix
    """
    x = df.values
    headers = list(df.columns.values)
    x_scaled = preprocessing.scale(x)
    df = pd.DataFrame(x_scaled, columns = headers)
    return df
    
    
def variance_columns(df_norm):
    """
    Input: normalized dataframe
    Output: dataframe without zero variance column
    """
    headers = list(df_norm.columns.values)
    selector = VarianceThreshold()
    x = df_norm.values
    x = selector.fit_transform(df_norm)
    indexlist = selector.get_support(indices = True)
    headers1 = []
    for i in range(0, len(indexlist)):
        indexs = indexlist[i]
        term = headers[indexs]
        headers1.append(term)
    df = pd.DataFrame(x, columns = headers1)
    return df
    
if __name__ == "__main__": 
    
    # read dataframe
    df = pd.read_pickle('data/final_user_features.pkl')
    df.reset_index(inplace=True) 
    
    # delete non numeric columns
    #del df['image_id']
    del df['index']
    del df['user_id']
    del df['outcome_P']
    del df['outcome_E']
    del df['outcome_M']
    del df['outcome_R']
    del df['outcome_A']
    del df['outcome_PERMA']
    
    
    print "Amount of features before variance thresholding: ", len(list(df.columns.values))
    # correlation matrix
    corr_matrix = correlation_matrix(df)
    
    # plot correlation matrix
    plot_correlation(corr_matrix)
    
    # normalize for variance thresholding
    df_norm = normalize_df(df)

    # variance threshold input has to be normalized
    df_var = variance_columns(df_norm)
    print "Amount of features after variance thresholding: ", len(list(df.columns.values))
