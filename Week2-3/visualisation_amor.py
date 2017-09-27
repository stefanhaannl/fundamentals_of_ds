# Import all the libraries required
import os
import json
import random
import time
import matplotlib
import numpy as np
import pandas as pd
#import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pprint as pp
import matplotlib.cm as cm
import numpy.random as nprnd
from scipy.interpolate import spline
from scipy.stats import gaussian_kde
from matplotlib.colors import rgb2hex
from descartes import PolygonPatch
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Point

import scipy.stats
# In[0]:
def load_pandas(filepath):
    """
    Specify the path, this function returns the pandas dataframe.
    """
    return pd.read_pickle(filepath)
    
def data_select(df_tweetjes, df_trump, topic):
    
    trump_tweets = df_trump.loc[df_trump.topic == topic]
    tweetjes = df_tweetjes.loc[df_tweetjes.topic == topic]
    
    tweetjes_time = tweetjes.sort_values('datetime')
    trump_time = trump_tweets.sort_values('datetime')
    
    
    tweetjes_time = tweetjes.reset_index()
    trump_time = trump_time.reset_index()
      
    
    return tweetjes_time, trump_time


def create_count_interval(tweetjes_time, trump_time, plot_hour):
    
    
    input_time = tweetjes_time.datetime.min()                # Starting time
    einde_der_tijden = tweetjes_time.datetime.max()     # End time
    first_tweet = tweetjes_time.datetime.loc[tweetjes_time.datetime == input_time]
    last_tweet = first_tweet + pd.DateOffset(hours = plot_hour)
    
    interval_counts = []
    interval_datetime = []
    interval_sentiment = []
    interval = pd.DataFrame()
    # Create time intervals and calculate amount of tweets per interval     
    while (last_tweet.iloc[0] < einde_der_tijden):
        last_tweet = first_tweet + pd.DateOffset(hours = plot_hour)
        
        mask = (tweetjes_time.datetime >= first_tweet.squeeze()) & (tweetjes_time.datetime < last_tweet.squeeze())
        interval_tweets = tweetjes_time.loc[mask]
        if len(interval_tweets)>0 : 
        #assign info
            interval_counts.append(len(interval_tweets))
            interval_sentiment.append(interval_tweets.sentiment.mean())
            time_df = first_tweet + pd.DateOffset(hours = (plot_hour/2))
            interval_datetime.append(time_df.iloc[0])
        
        else:
            interval_counts.append(0)
            interval_sentiment.append(0.5)
            time_df = first_tweet + pd.DateOffset(hours = (plot_hour/2))
            interval_datetime.append(time_df.iloc[0])
        first_tweet = last_tweet
    
    interval = pd.DataFrame({'counts' : interval_counts, 'sentiment' : interval_sentiment, 'datetime' : interval_datetime})
#    interval = pd.DataFrame({'counts' : interval_counts, 'datetime' : interval_datetime})
    
    return interval

# In[]
    
def plot_twifluence(df_tweetjes, df_trump, topic, plot_hour, plot_senti):
    # Retrieve tweets concerning topic & count amount of tweets/ average sentiment 
    # for pre-determined intervals between input/output time
    tweetjes_time, trump_time = data_select(df_tweetjes, df_trump, topic)
    
    input_time1 = tweetjes_time.datetime.min()                # Starting time
    end_time1 = tweetjes_time.datetime.max()     # End time
    
    # tump tweets zit ertussen
    mask1 = (trump_time.datetime >= input_time1) & (trump_time.datetime <= end_time1 )
    trump_time = trump_time.loc[mask1]
    
    input_time2 = trump_time.datetime.min() 
    index_begin2 = tweetjes_time.index[tweetjes_time['datetime'] < input_time2].tolist()
    index_begin1 = index_begin2[::-1]
    index_begin = index_begin1[650]
    
    end_time2 = trump_time.datetime.max() 
    index_end2 = tweetjes_time.index[tweetjes_time['datetime'] > input_time2].tolist()
    index_end = index_end2[3800]
    
    tweetjes_time = tweetjes_time.iloc[index_begin: index_end]
    pp.pprint(len(tweetjes_time))
    
    interval = create_count_interval(tweetjes_time, trump_time, plot_hour)
    
         
    # Interpolate count or sentiment data
    xnum = np.linspace(0, 10, num= len(interval.counts), endpoint = True)
    xnew = np.linspace(0, 10, num= (len(interval.counts)*10), endpoint=True)
    
    if plot_senti == 1:
        power_smooth = spline(xnum, interval.sentiment, xnew)
    else:
        power_smooth = spline(xnum, interval.counts, xnew)
    
    # Create new timeline to plot against (because of interpolation)
    
    begintime = matplotlib.dates.date2num(interval.datetime.iloc[0])
    endtime = matplotlib.dates.date2num(interval.datetime.iloc[-1])
    numdates = np.linspace(begintime, endtime, num = (len(interval.counts)*10), endpoint = True)
    
    interp_dates = matplotlib.dates.num2date(numdates)
    
    # Plot figure
    plt.plot(interp_dates,power_smooth, 'r-')
    #    plt.xticks(interp_dates)
    for k in trump_time .datetime.index:
        trump_date_number = matplotlib.dates.date2num(trump_time .datetime[k])
        min_diff = min(abs(i - trump_date_number) for i in numdates)
        toegevoegd = trump_date_number + min_diff
        afgenomen = trump_date_number - min_diff
        plus = [i for i,x in enumerate(numdates) if x == toegevoegd]
        minus = [i for i,x in enumerate(numdates) if x == afgenomen]
        if plus:
            hoogte = power_smooth[plus[0]]
        else:
            hoogte = power_smooth[minus[0]]
        
        plt.plot([trump_time .datetime[k],trump_time .datetime[k]],[0, hoogte], 'k-')
        
    
    if plot_senti == 1:
        plt.axis((begintime,endtime,0.4,1))
        plt.xlabel('Date: 08/31/2016 - 01/09/2016')
        plt.ylabel("Average sentiment on twitter (1 positive, 0 negative)")
        plt.title("Sentiment on Twitter over time about the topic: 'Wall Mexico'")
    else:
        plt.xlabel('Date')
        plt.ylabel('Amount of tweets')
        plt.title("Amount of tweets over time about the topic: 'Wall Mexico'")
    plt.show()
    
    return interval

def plot_results(topic, interval_length, sentiment_bool ):
    # Load all tweet data
    #    filepath = r'C:\Users\daniel\Downloads\true_tweets.pkl'
    #    df_tweetjes = load_pandas(filepath)
    
    # load trump tweet data
    filepath = r'final_dataset.pkl'
    filepath_trump = r'trump_df_final.pkl'
    df_tweetjes = load_pandas(filepath)
    df_trump = load_pandas(filepath_trump)
    
    
    # adjust trumps timeframe to original timeframe
    
    plot_twifluence(df_tweetjes, df_trump, topic, interval_length, sentiment_bool)
    

if __name__ == "__main__":
    
    # select data of interest and determine bin length (in hour)
    topic = 'Wall Mexico'               # topic
    interval_length = 8               # Interval length (sentiment 4, count)
    sentiment_bool = 0               # Plot sentiment(1) or count(0)
    
    plot_results(topic, interval_length, sentiment_bool)
