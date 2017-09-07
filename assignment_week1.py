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

#S_DIR is the directory in which your converted name2.shp file is located 
def projection(coordinates, extra_latitude, extra_longitude, shape):
    
    if shape == "Polygon":
        j=0
        for j in range(j,len(coordinates)):
            a = coordinates[j]
            i=0
            for i in range(i, len(a)):
                c = a[i]
                c[1] = c[1] + extra_longitude
                c[0] = c[0] + extra_latitude
                coordinates[j][i] = c
    else:
        i=0
        for i in range(i, len(coordinates)):
            c = coordinates[i]
            c[1] = c[1] + extra_longitude
            c[0] = c[0] + extra_latitude
            coordinates[i] = c
                    
# Scaling of states              
def scale(coordinates, coordinates2, scalefactor, shape):
    longcoords = []
    latcoords =[]
    if shape == "Polygon":
        j=0
        for j in range(j,len(coordinates)):
            a = coordinates[j]
            i=0
            for i in range(i, len(a)):
                c = a[i]
                longcoords.append(c[1])
                latcoords.append(c[0])
        minlong = min(longcoords)
        maxlong = max(longcoords)
        minlat = min(latcoords)
        maxlat = max(latcoords)
        
        j=0
        for j in range(j,len(coordinates)):
            a = coordinates[j]
            i=0
            for i in range(i, len(a)):
                c = a[i]
                c[1] = c[1] + scalefactor * (float(c[1]-minlong)/float(maxlong-minlong)-0.5)
                c[0] = c[0] + scalefactor * (float(c[0]-minlat)/float(maxlat-minlat)-0.5)
                coordinates[j][i] = c
    else:
        j=0
        for j in range(j,len(coordinates2)):
            a = coordinates2[j]
            k=0
            for k in range(k,len(a)):
                b = a[k]
                i=0
                for i in range(i, len(b)):
                    c = b[i]
                    longcoords.append(c[1])
                    latcoords.append(c[0])
            minlong = min(longcoords)
            maxlong = max(longcoords)
            minlat = min(latcoords)
            maxlat = max(latcoords)
        
        i=0
        for i in range(i, len(coordinates)):
            c = coordinates[i]
            
            scaler1 = float(scalefactor) * (float(c[1]-minlong)/float(maxlong-minlong)-0.5)
            c[1] = c[1] + scaler1
            c[0] = c[0] + scalefactor * (float(c[0]-minlat)/float(maxlat-minlat)-0.5)
            coordinates[i] = c
               
    return coordinates

def convert_to_color(average_age, min_age, max_age):
    alpha = float(average_age-min_age)/float((max_age-min_age))    
    return alpha

def move_state(state_name, extra_lat, extra_long):
    if geometry['type'] == 'Polygon':
            poly = geometry
            print "XXXXX"
            poly2 = poly.copy()
            coordinates = poly['coordinates']
            projected_coordinates = projection(coordinates,extra_lat, extra_long, shape="Polygon")
            poly2['coordinates'] = projected_coordinates 
            ax.add_patch(PolygonPatch(poly2, fc=RED,  alpha=gradient,  zorder=2))
                
    else:
             for polygon in geometry['coordinates'][0]:
                coordinates = polygon
                projected_coordinates = projection(coordinates,extra_lat, extra_long, shape="Multipolygonon")
                poly = Polygon(projected_coordinates)
                ax.add_patch(PolygonPatch(poly, fc=RED,  alpha=gradient,  zorder=2))
                
def scale_state(state_name):
    if geometry['type'] == 'Polygon':
            poly = geometry
            print "XXXXX"
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
                

if __name__ == "__main__": 
    population_data = pd.read_csv('pop_data/sc-est2016-agesex-civ.csv')
    population_data_all = population_data[population_data['SEX']==0]
    population_data_all = population_data_all[population_data_all['AGE']!=999]
    # Calculate average age of each state for the year 2016
    population_data_all['WEIGHT'] = population_data_all['AGE']*population_data_all['POPEST2016_CIV']
    avgAge = population_data_all.groupby('NAME')['WEIGHT'].sum() / population_data_all.groupby('NAME')['POPEST2016_CIV'].sum()
    
    max_AvgAge = max(avgAge)
    min_AvgAge  = min(avgAge)

    S_DIR = 'shapefiles/' 
    BLUE = '#5599ff'
    RED = '#F03911'
    BLACK = '#0B0B0B'
    
    extra_long = -20
    extra_lat = 20
    state = 'Hawaii'
    scalefactor = 1000

    with open(os.path.join(S_DIR, 'states.geojson')) as rf:    
        data = json.load(rf)

    fig = plt.figure() 
    ax = fig.gca()
    
    
    for feature in data['features']:
        geometry = feature['geometry']
        properties = feature['properties']
        state_name = properties['STATE_NAME']
        avgAge_state = avgAge[state_name]
        gradient = convert_to_color(avgAge_state,min_AvgAge, max_AvgAge)
        
        if state_name == state:
            #move_state(state, extra_lat,extra_long)
            #scale_state(state)
            pass
        else:
            if geometry['type'] == 'Polygon':
                # polygons
                poly = geometry
                ax.add_patch(PolygonPatch(poly, fc=BLACK,  alpha=gradient,  zorder=2))
            else:
                # multiple polygons
                for polygon in geometry['coordinates'][0]:
                    poly = Polygon(polygon)
                    ax.add_patch(PolygonPatch(poly, fc=BLACK, alpha=gradient, zorder=2))
    
    ax.axis('scaled')
    plt.axis('off')
    plt.show()
        

