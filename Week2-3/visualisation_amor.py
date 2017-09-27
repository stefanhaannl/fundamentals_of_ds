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
    
#    trump_tweets = df_trump[df_trump.topic == topic]
#    tweetjes = df_tweetjes[df_tweetjes.topic == topic]
    
    tweetjes_time = df_tweetjes.sort_values('datetime')
    trump_time = df_trump.sort_values('datetime')
    
    return tweetjes_time, trump_time
#    df_tweetjes.time[df_tweetjes.time > df_tweetjes.time[1]]


def create_count_interval(tweetjes_time, trump_time, plot_hour, input_time, einde_der_tijden):
    
    first_tweet = tweetjes_time.datetime.loc[tweetjes_time.datetime == input_time]
    last_tweet = first_tweet + pd.DateOffset(hours = plot_hour)
    
    interval_counts = []
    interval_datetime = []
    interval_sentiment = []
    interval = pd.DataFrame()
    #print last_tweet.squeeze()
    #print einde_der_tijden
# Create time intervals and calculate amount of tweets per interval     
    while (last_tweet.squeeze() < einde_der_tijden):
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
    
def plot_twifluence(df_tweetjes, df_trump, topic, plot_hour, input_time, einde_der_tijden, plot_senti):
    # Retrieve tweets concerning topic & count amount of tweets/ average sentiment 
    # for pre-determined intervals between input/output time
    tweetjes_time, trump_time = data_select(df_tweetjes, df_trump, topic)
    interval = create_count_interval(tweetjes_time, trump_time, plot_hour, input_time, einde_der_tijden)
    
         
    # Interpolate count or sentiment data
    xnum = np.linspace(0, 10, num= len(interval.counts), endpoint = True)
    xnew = np.linspace(0, 10, num= (len(interval.counts)*10), endpoint=True)
    
    if plot_senti == 1:
        power_smooth = spline(xnum, interval.sentiment, xnew)
    else:
        power_smooth = spline(xnum, interval.counts, xnew)
    
    # Create new timeline to plot against (because of interpolation)
    
    # waarom [0][1]
    begintime = matplotlib.dates.date2num(interval.datetime.iloc[0])
    endtime = matplotlib.dates.date2num(interval.datetime.iloc[-1])
    numdates = np.linspace(begintime, endtime, num = (len(interval.counts)*10), endpoint = True)
    
    interp_dates = matplotlib.dates.num2date(numdates)
    
    # Plot figure
    plt.plot(interp_dates,power_smooth, 'r-')
#    plt.xticks(interp_dates)
    for k in df_trump.datetime.index:
        trump_date_number = matplotlib.dates.date2num(df_trump.datetime[k])
        min_diff = min(abs(i - trump_date_number) for i in numdates)
        toegevoegd = trump_date_number + min_diff
        afgenomen = trump_date_number - min_diff
        plus = [i for i,x in enumerate(numdates) if x == toegevoegd]
        minus = [i for i,x in enumerate(numdates) if x == afgenomen]
        if plus:
            hoogte = power_smooth[plus[0]]
        else:
            hoogte = power_smooth[minus[0]]
        
        plt.plot([df_trump.datetime[k],df_trump.datetime[k]],[power_smooth.min(), hoogte+(hoogte/5)], 'k-')

    plt.show()
    
    return interval

def plot_results()
	# Load all tweet data
	#    filepath = r'C:\Users\daniel\Downloads\true_tweets.pkl'
	#    df_tweetjes = load_pandas(filepath)
    
    # load trump tweet data
    filepath = r'trump_df_sentiment.pkl'
    df_tweetjes = load_pandas(filepath)
    df_trump = df_tweetjes[0:100]

    
    # select data of interest and determine bin length (in hour)
    topic = 'healthcare'                # topic
    plot_hour = 25               # Plotting Interval length
    
    # Statistical parameters
    stat_hour = 48                      # Hours minus and plus trump time (to compare the two intervals)
    
    # Plotting parameters
    input_time = df_tweetjes.datetime.min()                # Starting time
    einde_der_tijden = df_tweetjes.datetime.max()     # End time
    plot_senti = 0                                    # Plot sentiment or count
                                                            # 1 = true, 0 = false
    
    plot_twifluence(df_tweetjes, df_trump, topic, plot_hour, input_time, einde_der_tijden, plot_senti)
    

if __name__ == "__main__":
	plot_results()
