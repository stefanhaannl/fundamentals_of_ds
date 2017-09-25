# Import all the libraries required
import os
import json
import random
import time
import matplotlib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pprint as pp
import matplotlib.cm as cm
import numpy.random as nprnd
from scipy.interpolate import spline
from scipy.stats import gaussian_kde
from matplotlib.colors import rgb2hex
from descartes import PolygonPatch
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Point

# In[0]:
def load_pandas():
    """
    Specify the path, this function returns the pandas dataframe.
    """
    pandasfilepath = r'C:\Users\daniel\Downloads\true_tweets.pkl'
    return pd.read_pickle(pandasfilepath)
    
# In[1]:
# Import dataset
def random_data():
    import random
    n = 10000
    i = 0
    tweetjes = {}
    topics = ['immigration','taxes','jobs','healthcare']

    for i in range(i, n):
        rand_lon = -random.randint(70, 120)
        rand_lat = random.randint(25, 50)    
        rand_topic = random.randint(0,3)
        topic = topics[rand_topic]
        time = randomDate("1/1/2008 1:30 PM", "2/1/2008 4:50 AM", random.random())
        tweetjes[i] = {'longitude': rand_lon, 'latitude': rand_lat, 'topic': topic, 'time':time}
        
    df_tweetjes = pd.DataFrame.from_dict(tweetjes, orient='index')
    df_tweetjes.sentiment = nprnd.randint(1, size=len(df_tweetjes))
    
    return df_tweetjes

def strTimeProp(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def randomDate(start, end, prop):
    return strTimeProp(start, end, '%m/%d/%Y %I:%M %p', prop)

def specific_tweet(df_tweetjes, trump_tweet):
    a = 1
    return a

# In[]

def data_select(df_tweetjes, df_trump, topic):
    
#    trump_tweets = df_trump[df_trump.topic == topic]
#    tweetjes = df_tweetjes[df_tweetjes.topic == topic]
    
    tweetjes_time = df_tweetjes.sort_values('datetime')
    trump_time = df_trump.sort_values('datetime')
    
    return tweetjes_time, trump_time
#    df_tweetjes.time[df_tweetjes.time > df_tweetjes.time[1]]


def create_count_interval(tweetjes_time, trump_time, input_time, einde_der_tijden):
    
    first_tweet = tweetjes_time.datetime.loc[tweetjes_time.datetime == input_time]
    last_tweet = first_tweet + pd.DateOffset(hours = hour)
   # interval = pd.DataFrame(columns = ['counts','datetime'])
    interval_counts = []
    interval_datetime = []
    interval_sentiment = []
    interval = pd.DataFrame()
    
# Create time intervals and calculate amount of tweets per interval     
    while (last_tweet.squeeze() < einde_der_tijden):
        last_tweet = first_tweet + pd.DateOffset(hours = hour)
        
        mask = (tweetjes_time.datetime >= first_tweet.squeeze()) & (tweetjes_time.datetime < last_tweet.squeeze())
        interval_tweets = tweetjes_time.loc[mask]
        
        #assign info
        interval_counts.append(len(interval_tweets))
        interval_sentiment.append(interval_tweets.sentiment.mean())
        interval_datetime.append(first_tweet + pd.DateOffset(hours = (hour/2)))
        
        first_tweet = last_tweet
    
    interval = pd.DataFrame({'counts' : interval_counts, 'sentiment' : interval_sentiment, 'datetime' : interval_datetime})
    return interval

# In[]
    
def plot_twifluence(df_tweetjes, df_trump, topic, hour, input_time, einde_der_tijden, plot_senti):
    
    # Retrieve tweets concerning topic & count amount of tweets/ average sentiment 
    # for pre-determined intervals between input/output time
    tweetjes_time, trump_time = data_select(df_tweetjes, df_trump, topic)
    interval = create_count_interval(tweetjes_time, trump_time, input_time, einde_der_tijden)
        
    # Interpolate count or sentiment data
    xnum = np.linspace(0, 10, num= len(interval.counts), endpoint = True)
    xnew = np.linspace(0, 10, num= (len(interval.counts)*10), endpoint=True)
    if plot_senti == 1:
        power_smooth = spline(xnum, interval.sentiment, xnew)
    else:
        power_smooth = spline(xnum, interval.counts, xnew)
    
    # Create new timeline to plot against (because of interpolation)
    begintime = matplotlib.dates.date2num(interval.datetime[0][1])
    endtime = matplotlib.dates.date2num(interval.datetime[len(interval.datetime)-1][1])
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

#############################################################################
#---------------------------------------------------------------------------#
#############################################################################
# In[]

if __name__ == "__main__":
    
    # Load tweet data
    df_tweetjes = load_pandas()

    # load trumps tweets
    # df_trump = ....

    ######### TIJDELIJK ###########
    df_trump = df_tweetjes[20000:20010]
    
    
    # select data of interest and determine bin length (in hour)
    topic = 'healthcare'                                    # topic
    hour = 0.1                                              # Interval length
    input_time = df_tweetjes.datetime.min()                 # Starting time
    einde_der_tijden = df_tweetjes.datetime.iloc[50000]     # End time
    plot_senti = 0                                          # Plot sentiment
                                                            # 1 = true, 0 = false
    
    # Wow zie deze woordspeling twitter + influence
    plot_twifluence(df_tweetjes, df_trump, topic, hour, input_time, einde_der_tijden, plot_senti)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# In[]    
    # Get population data
    population_data = pd.read_csv('pop_data/sc-est2016-agesex-civ.csv')
    population_data_all = population_data[population_data['SEX']==0]
    population_data_all = population_data_all[population_data_all['AGE']!=999]

    # Sum the population of each state for each year on the dataset 'population_data_all'
    population_data_all.groupby(by=['NAME'], as_index=False)[['POPEST2010_CIV','POPEST2011_CIV','POPEST2012_CIV',
                                            'POPEST2013_CIV','POPEST2014_CIV','POPEST2015_CIV','POPEST2016_CIV']].sum()

    # Calculate average age of each state for the year 2016
    population_data_all['WEIGHT'] = population_data_all['AGE']*population_data_all['POPEST2016_CIV']
    avgAge = population_data_all.groupby('NAME')['WEIGHT'].sum() / population_data_all.groupby('NAME')['POPEST2016_CIV'].sum()
     