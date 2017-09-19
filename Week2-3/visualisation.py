# Import all the libraries required
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pprint as pp
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex
from descartes import PolygonPatch
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Point
import pandas as pd
import numpy as np

# In[0]:
def load_pandas():
    """
    Specify the path, this function returns the pandas dataframe.
    """
    pandasfilepath = r'C:\Users\shaan\Documents\true_tweets.pkl'
    return pd.read_pickle(pandasfilepath)
    
# In[1]:
# Import dataset

import random
n = 10000
i = 0
tweetjes = {}

for i in range(i, n):
    rand_lon = -random.randint(70, 120)
    rand_lat = random.randint(25, 50)    
    name = 'name'
    tweetjes[i] = {'longitude': rand_lon, 'latitude': rand_lat, 'name': name}
    
df_tweetjes = pd.DataFrame.from_dict(tweetjes, orient='index')


# In[2]

def sum_tweets(df_tweetjes, S_DIR, Fname):
    # Sum total amount of tweets per state
    us = gpd.read_file(S_DIR+Fname)

    tweet_point = [Point(xy) for xy in zip(df_tweetjes.longitude, df_tweetjes.latitude)]

    geo_tweet = gpd.GeoDataFrame({"geometry": tweet_point, "name": df_tweetjes["name"]})

    # Set coordinate system (WGS 84)
    us.crs = {'init': 'epsg:4326'}
    geo_tweet.crs = {'init': 'epsg:4326'}

    # Calculate intersections per state
    us_tweets = gpd.tools.sjoin(geo_tweet, us, how = "right", op = 'intersects')
    state_tweets = us_tweets.groupby("STATE_NAME").size()
    
    return state_tweets


def per_inhabitant(population_data_all, state_tweets):
    # Divide by inhabitants and multiply by 1000
    state_pop = population_data_all.groupby('NAME')['POPEST2016_CIV'].sum()
    state_tweets_1000 = (state_tweets/state_pop) * 1000

    return state_tweets_1000

# In[]

def get_alpha(value_state, minimum, maximum):
    """
    Get an alpha value based on the age range.
    Inputs: average_age, min_age, max_age
    Outputs: floating point number between 0 and 1
    """    
    alpha = float(value_state-minimum)/float((maximum-minimum))
    
    return alpha

def alphatohex(alpha): 
    """
    Converts a number between 0 and 1 to a interpolate of red and blue
    Inputs: alpha value between 0 and 1
    Outputs: color code (string)
    """
    #c1 = (0,0,255) #blue
    #c2 = (0,255,0) #green
    if alpha >= 0.5:
        b = 255
        g = int(255-round(510*(alpha-0.5)))
    if alpha <= 0.5:
        g = 255
        b = int(round(510*(alpha)))
    rgb = (b,0,g)
    return "#"+"".join(map(chr, rgb)).encode('hex')



# Reproject function
# Geometry contains all the coordinates that determine the states outline. Coordinates are stored as pairs that 
# represent locations. Sets of locations form polygon outlines, between which lines are drawn.
# Some states consist out of more than one polygon, which are stored as mulitpolygons consisting out of polygon sets

def projection(coordinates, extra_latitude, extra_longitude, shape):
    # States consisting out of one polygon
    if shape == "Polygon":
        print
        j=0
        for j in range(j,len(coordinates)): 
            a = coordinates[j]
            i=0
            for i in range(i,len(a)):
                c = a[i]
                c[1] = c[1] + extra_longitude
                c[0] = c[0] + extra_latitude
                coordinates[j][i] = c
                #print coordinates[j][i]
    else:
        # States consisting out of multiple polygons
        i=0
        for i in range(i,len(coordinates)):
            c = coordinates[i]
            c[1] = c[1] + extra_longitude
            c[0] = c[0] + extra_latitude
            coordinates[i] = c


    return coordinates
# In[ ]:

def scale(coordinates, scalefactor, shape, minlong,maxlong, minlat, maxlat):
    
    if shape == "Polygon":
        # Find the minimum and maximum longitude and latitude of a state outline
        
        longcoords = []
        latcoords = []
        
        for j in range(len(coordinates)):
            a = coordinates[j]
            for i in range(len(a)):
                c = a[i]
                longcoords.append(c[1])
                latcoords.append(c[0])
        minlong = min(longcoords)
        maxlong = max(longcoords)
        minlat = min(latcoords)
        maxlat = max(latcoords)
        
        # Calculate new coordinate values based on states maximum and minimum long/latitude, causing proportional scaling
        # of every set of coordinates.
        # The -0.5 is implemented such that the middle stays in the same place. Coordinates above the states middle long/latitude
        # will extend in the opposite direction than the coordinates under this value.
        for j in range(len(coordinates)):
            a = coordinates[j]
            for i in range(len(a)):
                c = a[i]
                c[1] = c[1] + scalefactor * (float(c[1]-minlong)/float(maxlong-minlong)-0.5)
                c[0] = c[0] + scalefactor * (float(c[0]-minlat)/float(maxlat-minlat)-0.5)
                coordinates[j][i] = c
    else:
        #  Find the minimum and maximum longitude and latitude of the combined outline of a state with multiple polygons
        
        
        
        # Calculate new coordinate values based on these maximum and minimum long/lat values.
        i=0
        for i in range(i, len(coordinates)):
            c = coordinates[i]
            
            c[1] = c[1] + scalefactor * (float(c[1]-minlong)/float(maxlong-minlong)-0.5)
            c[0] = c[0] + scalefactor * (float(c[0]-minlat)/float(maxlat-minlat)-0.5)
            coordinates[i] = c
               
    return coordinates

def get_midden(coordinates2, shape):
    longcoords = []
    latcoords =[]
    
    if shape == 'Polygon':
        k=0
        for k in range(len(coordinates2)):
            b = coordinates2[k]
            i=0
            for i in range(len(b)):
                c = b[i]
                longcoords.append(c[1])
                latcoords.append(c[0])
            minlong = min(longcoords)
            maxlong = max(longcoords)
            minlat = min(latcoords)
            maxlat = max(latcoords)
    else:
        j=0
        for j in range(len(coordinates2)):
                a = coordinates2[j]
                k=0
                for k in range(len(a)):
                    b = a[k]
                    i=0
                    for i in range(len(b)):
                        c = b[i]
                        longcoords.append(c[1])
                        latcoords.append(c[0])
                minlong = min(longcoords)
                maxlong = max(longcoords)
                minlat = min(latcoords)
                maxlat = max(latcoords)
    return minlong, maxlong, minlat, maxlat

# Retrieve coordinates of given state, transform and plot on map
def scale_state(geometry, scalefactor, color,extra_lat, extra_long,ax):
    
    if geometry['type'] == 'Polygon':
        poly = geometry
        poly2 = poly.copy()
        coordinates = poly['coordinates']
        minlong, maxlong, minlat, maxlat = get_midden(coordinates,"Polygon")
        projected_coordinates = scale(coordinates,scalefactor, "Polygon",  minlong,maxlong, minlat,maxlat)
        projected_coordinates1 = projection(projected_coordinates,extra_lat, extra_long, "Polygon")
        poly2['coordinates'] = projected_coordinates1
        ax.add_patch(PolygonPatch(poly2, fc=color,  alpha=1,  zorder=2))
        
                
    else:
        totale_eilanden = geometry['coordinates']
        minlong, maxlong, minlat, maxlat = get_midden(totale_eilanden,"Multippolygon")
        j=0
        for j in range(j,len(totale_eilanden)):
            for polygon in geometry['coordinates'][j]:
                
                coordinates = polygon
                projected_coordinates = scale(coordinates, scalefactor, "Multipolygonon", minlong,maxlong, minlat,maxlat)
                projected_coordinates1 = projection(projected_coordinates,extra_lat, extra_long, "Multipolygonon")
                
                poly = Polygon(projected_coordinates1)
                ax.add_patch(PolygonPatch(poly, fc=color,  alpha=1,  zorder=2))

        
        
def move_state(geometry, extra_lat, extra_long, color):
    if geometry['type'] == 'Polygon':
        # States consisting out of one polygon
        poly = geometry
        poly2 = poly.copy()
        coordinates = poly['coordinates']
        projected_coordinates = projection(coordinates,extra_lat, extra_long, shape="Polygon")
        #pp.pprint(projected_coordinates)
        poly2['coordinates'] = projected_coordinates 
        ax.add_patch(PolygonPatch(poly2, fc=color,  alpha=1,  zorder=2))
                
    else:
        # States consisting out of multiple polygons
        
        for polygon in geometry['coordinates'][0]:
            coordinates = polygon
            projected_coordinates = projection(coordinates,extra_lat, extra_long, shape="Multipolygonon")
            poly = Polygon(projected_coordinates)
            ax.add_patch(PolygonPatch(poly, fc=color,  alpha=1,  zorder=2))
            
# In[ ]:

def plot_states(data, value, scalefactor, extra_lat, extra_long, scaled_state):
    
    # Retrieve maximum and minimum average age
    maximum = max(value)
    minimum  = min(value)
    
    #fig = plt.figure(figsize=(20,20)) 
    fig = plt.figure() 
    ax = fig.gca()
        
        
    for feature in data['features']:            # Retrieve each state (feature) from database
        geometry = feature['geometry']          # Retrieve geometry of state (dictionary with coordinates)
        properties = feature['properties']
        state_name = properties['STATE_NAME']
        value_state = value[state_name]
        alpha = get_alpha(value_state, minimum, maximum) # Determine fill color based on states average age
        color = alphatohex(alpha)
            
        if state_name in scaled_state:
            #move_state(geometry, extra_lat, extra_long, color)
            scale_state(geometry, scalefactor, color,extra_lat, extra_long,ax)
            
        #    pass
        else:
            if geometry['type'] == 'Polygon':
                # States consisting out of one polygon
                poly = geometry
                ax.add_patch(PolygonPatch(poly, fc=color,  alpha=1,  zorder=2))
            else:
                # States consisting out of multiple polygons
                totale_eilanden = geometry['coordinates']
                j=0
                for j in range(j,len(totale_eilanden)):
                    for polygon in geometry['coordinates'][j]:
                        poly = Polygon(polygon)
                        ax.add_patch(PolygonPatch(poly, fc=color, alpha=1, zorder=2))
            
    ax.axis('scaled')
    plt.axis('off')
    plt.show()
    
if __name__ == "__main__":
    
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
     

    # Set parameters  
    """
    scaled_state = scaled or state
    extra_long/lat = to move state (0,0) if not move
    scalefactor = to scale state (0) if not scaled
    """
    # S_DIR is directory & Fname = shapefile name
    S_DIR = 'shapefiles/'
    Fname = 'states.geojson'
    #BLUE = '#5599ff'
    #RED = '#F03911'
    BLACK = '#0B0B0B'
    GRAY = '#DCDCDC'

# Longitude latitudes staan nog verkeerd in de functions, hieronder staat het goed
    extra_lat = 10  # increase/decrease in latitude
    extra_long = 0     # increase/decrease in longitude
    scaled_state = ['Wyoming', 'Montana','North Dakota','South Dakota','Nebraska','Wisconsin','Iowa','Minnesota']    # State(s) that will be altered
    scalefactor = 0  # Increase of state size in degrees
        
        
    # open coordinates/data of stats
    with open(os.path.join(S_DIR, Fname)) as rf:    
        data = json.load(rf)
    
    # Group amount of tweets by state and calculate tweets / citizen
    state_tweets = sum_tweets(df_tweetjes, S_DIR, Fname)
    state_tweets_1000 = per_inhabitant(population_data_all,state_tweets)
    
    # plot states
    plot_states(data, state_tweets_1000, scalefactor, extra_long, extra_lat, scaled_state)
