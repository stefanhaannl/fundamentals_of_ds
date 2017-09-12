
# coding: utf-8

# <h1>Fundamentals of Data Science voor ons</h1>
# 
# <h2>Assignment 1</h2>
# 
# **GOAL** In this notebook we are going to cover the following practical aspects of data science:
# 
# - Reading a csv file and loading it to a dataframe in python using pandas library
# - Filtering out the required columns in the dataframe
# - Summarising data based on the fields. Ex: Summing up all the rows corresponding to a certain entry in the dataset
# - Plot shape of United States using the geographic data i.e. data with all the coordinates
# - Scale and move the states using data of the coordinates
# - Colour the states based on the average age of their population

# To complete this assignment you need to have a running Anaconda installation with Python 2.7 on your device. Python package prerequisites include:
# +  **Python Data Analysis Library** [Pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)  
# +  **GDAL - Geospatial Data Abstraction Library ** [GDAL](http://www.gdal.org/index.html)   
# 
# 

# **Pandas** is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python. Additionally, it has the broader goal of becoming the most powerful and flexible open source data analysis / manipulation tool available in any language. It is already well on its way toward this goal.
# 
# Pandas is well suited for many different kinds of data:
# 
# - Tabular data with heterogeneously-typed columns, as in an SQL table or Excel spreadsheet
# - Ordered and unordered (not necessarily fixed-frequency) time series data.
# - Arbitrary matrix data (homogeneously typed or heterogeneous) with row and column labels
# - Any other form of observational / statistical data sets. The data actually need not be labeled at all to be placed into a pandas data structure
# 
# The two primary data structures of pandas, Series (1-dimensional) and DataFrame (2-dimensional), handle the vast majority of typical use cases in finance, statistics, social science, and many areas of engineering. 

# In[2]:

# Import all the libraries required
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint as pp
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex
from descartes import PolygonPatch
from shapely.geometry import Polygon, MultiPolygon


# <h3>A look at the data</h3>
# Throughout this assignment, the dataset that we would be using will be the US population statistics 2010-2016. It can be downloaded from https://www2.census.gov/programs-surveys/popest/datasets/2010-2016/state/asrh/.

# As a first step, we will structure the data into a pandas DataFrame to simplify the data manipulation. We will start by loading the csv file of data into a DataFrame called population_data and we will then filter columns based on our use. *Take a look at the data and its fields*

# In[3]:

population_data = pd.read_csv('pop_data/sc-est2016-agesex-civ.csv')


# In[4]:

population_data.shape


# The complete dataset has 13,572 rows and 15 columns. It can be verified by looking at the shape of your dataframe by <i>population_data.shape</i>. To select specific columns, create a list of column names and view the dataframe as:

# In[5]:

# Filtering the data by the columns ['NAME','SEX','AGE','POPEST2016_CIV']
population_data[['NAME','SEX','AGE','POPEST2016_CIV']]


# In[6]:

# Filters out data for all sexes and all age group and store in a new dataframe 'population_data_all'
# 0 staat voor man en vrouw samen (dus niet apart gespecificeerd)
population_data_all = population_data[population_data['SEX']==0]
# Haalt alle values eruit waarvan een geldige leeftijd bekend is.
population_data_all = population_data_all[population_data_all['AGE']!=999]


# In[7]:

# Sum the population of each state for each year on the dataset 'population_data_all'
population_data_all.groupby(by=['NAME'], as_index=False)[['POPEST2010_CIV','POPEST2011_CIV','POPEST2012_CIV',
                                        'POPEST2013_CIV','POPEST2014_CIV','POPEST2015_CIV','POPEST2016_CIV']].sum()


# <h3> Installations </h3>
# GDAL - Geospatial Data Abstraction Library <br>
#     http://www.gdal.org/index.html
# 
# For example,
# ** On Linux Fedora:** 
#          <ul><i>yum install libpng</i>   
#          <i> yum install libtiff </i>  
#          <i> sudo dnf install gdal gdal-devel</i><br>
#          
# ** In Ubuntu:**
#          <ul><i>sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update</i>   
#          <i> sudo apt-get install gdal-bin </i>  
#          <i> To verify after installation, try: $ ogrinfo</i><br>
#          <i> If the installation was successful, you will see something like this:
# 
# + Usage: ogrinfo [--help-general] [-ro] [-q] [-where restricted_where]
#                [-spat xmin ymin xmax ymax] [-fid fid]
#                [-sql statement] [-al] [-so] [-fields={YES/NO}]
#                [-geom={YES/NO/SUMMARY}][--formats]
#                datasource_name [layer [layer ...]]
# </i><br>
#         
#                   

# **In Windows:**
#          Refer to https://sandbox.idre.ucla.edu/sandbox/tutorials/installing-gdal-for-windows

# <h3> Get a US states shapefile </h3>        
# Sources
# - ArcGIS shapefile of US 50 states + DC (https://www.arcgis.com/home/item.html?id=f7f805eb65eb4ab787a0a3e1116ca7e5)
# <ul>
# Recommended – convert to GeoJSON    
# (<i> ogr2ogr  -f  GeoJSON  [name1].geojson  [name2].shp </i> - where, 'name1' is the name of the downloaded geojson file and 'name2' is the name you want to specify for the shape file)

# In[11]:

#S_DIR is the directory in which your converted name2.shp file is located 
#S_DIR = 'US_shape/' 
S_DIR = 'shapefiles/'
BLUE = '#5599ff'
BLACK = '#000000'

with open(os.path.join(S_DIR, 'states.geojson')) as rf:    
    data = json.load(rf)

fig = plt.figure() 
ax = fig.gca()


# In[12]:

for feature in data['features']:
    geometry = feature['geometry']
    if geometry['type'] == 'Polygon':
        poly = geometry
        ax.add_patch(PolygonPatch(poly, fc=BLACK, ec=BLUE,  alpha=0.5, zorder=2))
    else:
        for polygon in geometry['coordinates'][0]:
            poly = Polygon(polygon)
            ax.add_patch(PolygonPatch(poly, fc=BLACK, ec=BLUE, alpha=0.5, zorder=2))

ax.axis('scaled')
plt.axis('off')
plt.show()


# **TASKS** <font color="red">
# <h3> Improve the map </h3>
# - Try a different projection (example: US Census Bureau shapefile)
# - Scale and move Alaska 
# - Increase the size of Hawaii
# - Color the map based on the average age of each state for the year 2016 
#     - Example: for average age between 65-70 years color the states in maroon, for 60-65 years in red and so on
# 
# <font color="blue">
# Something else: <b> Matplotlib Basemap Toolkit </b> <br>
# Even more: <b> GeoPandas </b>

# In[13]:

def get_alpha(average_age, min_age, max_age):
    """
    Get an alpha value based on the age range.
    Inputs: average_age, min_age, max_age
    Outputs: floating point number between 0 and 1
    """
    bottom = min_age
    upper = max_age
    
    alpha = float(average_age-min_age)/float((max_age-min_age))
    
    return alpha

def alphatohex(alpha): 
    """
    Converts a number between 0 and 1 to a interpolate of red and blue
    Inputs: alpha value between 0 and 1
    Outputs: color code (string)
    """
    c1 = (0,0,255)
    c2 = (0,255,0)
    if alpha >= 0.5:
        b = 255
        g = int(255-round(510*(alpha-0.5)))
    if alpha <= 0.5:
        g = 255
        b = int(round(510*(alpha)))
    rgb = (0,g,b)
    return "#"+"".join(map(chr, rgb)).encode('hex')


# <font color="black">
# <h3> Assignment </h3>
# 
# This assignment should result in a report in the format as mentioned in blackboard. 
# 
# <h4>The deadline for the assignment is <u> <font color="red"> **13-09-2017 23:59PM CEST. **</u> </h4>
# 

# In[14]:

# Reproject function
# Geometry contains all the coordinates that determine the states outline. Coordinates are stored as pairs that 
# represent locations. Sets of locations form polygon outlines, between which lines are drawn.
# Some states consist out of more than one polygon, which are stored as mulitpolygons consisting out of polygon sets

def projection(coordinates, extra_latitude, extra_longitude, shape):
    # States consisting out of one polygon
    if shape == "Polygon":
        for j in range(len(coordinates)):
            a = coordinates[j]
            for i in range(len(a)):
                c = a[i]
                c[1] = c[1] + extra_longitude
                c[0] = c[0] + extra_latitude
                coordinates[j][i] = c
    else:
        # States consisting out of multiple polygons
        for i in range(len(coordinates)):
            c = coordinates[i]
            c[1] = c[1] + extra_longitude
            c[0] = c[0] + extra_latitude
            coordinates[i] = c

def move_state(state_name, extra_lat, extra_long):
    if geometry['type'] == 'Polygon':
        # States consisting out of one polygon
            poly = geometry
            poly2 = poly.copy()
            coordinates = poly['coordinates']
            projected_coordinates = projection(coordinates,extra_lat, extra_long, shape="Polygon")
            poly2['coordinates'] = projected_coordinates 
            ax.add_patch(PolygonPatch(poly2, fc=RED,  alpha=gradient,  zorder=2))
                
    else:
    # States consisting out of multiple polygons
            for polygon in geometry['coordinates'][0]:
                coordinates = polygon
                projected_coordinates = projection(coordinates,extra_lat, extra_long, shape="Multipolygonon")
                poly = Polygon(projected_coordinates)
                ax.add_patch(PolygonPatch(poly, fc=RED,  alpha=gradient,  zorder=2))


# In[15]:

def scale(coordinates, coordinates2, scalefactor, shape):
    longcoords = []
    latcoords =[]
    
    if shape == "Polygon":
        # Find the minimum and maximum longitude and latitude of a state outline
        
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
        for j in range(len(coordinates2)):
            a = coordinates2[j]
            for k in range(len(a)):
                b = a[k]
                for i in range(len(b)):
                    c = b[i]
                    longcoords.append(c[1])
                    latcoords.append(c[0])
            minlong = min(longcoords)
            maxlong = max(longcoords)
            minlat = min(latcoords)
            maxlat = max(latcoords)
        
        # Calculate new coordinate values based on these maximum and minimum long/lat values.
        for i in range(i, len(coordinates)):
            c = coordinates[i]
            
            c[1] = c[1] + scalefactor * (float(c[1]-minlong)/float(maxlong-minlong)-0.5)
            c[0] = c[0] + scalefactor * (float(c[0]-minlat)/float(maxlat-minlat)-0.5)
            coordinates[i] = c
               
    return coordinates

# Retrieve coordinates of given state, transform and plot on map
def scale_state(state_name):
    if geometry['type'] == 'Polygon':
        poly = geometry
        poly2 = poly.copy()
        coordinates = poly['coordinates']
        projected_coordinates = scale(coordinates,_, scalefactor, shape="Polygon")
        poly2['coordinates'] = projected_coordinates 
        ax.add_patch(PolygonPatch(poly2, fc=RED,  alpha=gradient,  zorder=2))
                
    else:
        for polygon in geometry['coordinates'][0]:
            coordinates = polygon
            coordinates2 = geometry['coordinates']
            projected_coordinates = scale(coordinates, coordinates2, scalefactor, shape="Multipolygonon")
            poly = Polygon(projected_coordinates)
            ax.add_patch(PolygonPatch(poly, fc=RED,  alpha=gradient,  zorder=2))


# In[16]:

# Calculate average age of each state for the year 2016
population_data_all['WEIGHT'] = population_data_all['AGE']*population_data_all['POPEST2016_CIV']
avgAge = population_data_all.groupby('NAME')['WEIGHT'].sum() / population_data_all.groupby('NAME')['POPEST2016_CIV'].sum()
 
# Retrieve maximum and minimum average age
max_AvgAge = max(avgAge)
min_AvgAge  = min(avgAge)


# In[20]:

# Set parameters  
S_DIR = 'shapefiles/' 
BLUE = '#5599ff'
RED = '#F03911'
BLACK = '#0B0B0B'

extra_long = -20  # increase/decrease in longitude
extra_lat = 20     # increase/decrease in latitude
state = 'Iowa'    # State(s) that will be altered
scalefactor = 100  # Increase of state size in degrees
    
#-----------------------------------------------------------------#    
    
with open(os.path.join(S_DIR, 'states.geojson')) as rf:    
    data = json.load(rf)

fig = plt.figure() 
ax = fig.gca()
    
    
for feature in data['features']:            # Retrieve each state (feature) from database
    geometry = feature['geometry']          # Retrieve geometry of state (dictionary with coordinates)
    properties = feature['properties']
    state_name = properties['STATE_NAME']
    avgAge_state = avgAge[state_name]
    alpha = get_alpha(avgAge_state,min_AvgAge, max_AvgAge) # Determine fill color based on states average age
    color = alphatohex(alpha)
        
    if state_name == state:
        #move_state(state, extra_lat, extra_long)
        #scale_state(state)
        pass
    else:
        if geometry['type'] == 'Polygon':
            # States consisting out of one polygon
            poly = geometry
            ax.add_patch(PolygonPatch(poly, fc=color,  alpha=1,  zorder=2))
        else:
            # States consisting out of multiple polygons
            for polygon in geometry['coordinates'][0]:
                poly = Polygon(polygon)
                ax.add_patch(PolygonPatch(poly, fc=color, alpha=1, zorder=2))
    
ax.axis('scaled')
plt.axis('off')
plt.show()

