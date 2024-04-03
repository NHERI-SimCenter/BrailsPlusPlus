# Written: fmk, 3/24
#  guts of code (fetch_roi and bbox_poly) provided bacentinar 3/24

import math
import json
import requests
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import groupby
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, box
from shapely.ops import linemerge, unary_union, polygonize
from shapely.strtree import STRtree
from brails.utils.geoTools import *
import concurrent.futures
from requests.adapters import HTTPAdapter, Retry
import unicodedata

class RegionBoundary:
    """
    A class representing to obtain the boounding polygon for a region. 

    Attributes:


     Methods:
        __init__: Constructor validates input
        get_boundary(): returns the boundary polygon of the region provided
    """
        
    def __init__(self, input :dict = None):
       
        """        
        Check inputs

        Args:
          data (dict): The data to be checked
        """
        
        self.data = {}
        valid_data = False
        if isinstance(input, dict):
            if 'type' in input:
                valid_data = True
                self.type = input['type']
                # some more checks

        if valid_data == True:
            self.data = input['data']
        else:
            self.data = {}

    def __fetch_roi(self, queryarea, outfile=False):

        #
        # Search for the query area using Nominatim API:
        #
        
        print(f"\nSearching for {queryarea}...")
        queryarea = queryarea.replace(" ", "+").replace(',','+')
        
        queryarea_formatted = ""
        for i, j in groupby(queryarea):
            if i=='+':
                queryarea_formatted += i
            else:
                queryarea_formatted += ''.join(list(j))
        
        nominatimquery = ('https://nominatim.openstreetmap.org/search?' +
                          f"q={queryarea_formatted}&format=jsonv2") 
        headers = {'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)'+
                                  ' AppleWebKit/537.36 (KHTML, like Gecko)'+
                                  ' Chrome/39.0.2171.95 Safari/537.36')}               
        r = requests.get(nominatimquery, headers=headers)
        datalist = r.json()
        
        areafound = False
        for data in datalist:
            queryarea_osmid = data['osm_id']
            queryarea_name = data['display_name']
            if data['osm_type']=='relation':
                areafound = True
                break
        
        if areafound==True:
            try:
                print(f"Found {queryarea_name}")
            except:
                queryareaNameUTF = unicodedata.normalize(
                    'NFKD', queryarea_name).encode('ascii', 'ignore')
                queryareaNameUTF = queryareaNameUTF.decode("utf-8")
                print(f"Found {queryareaNameUTF}") 
        else:
            sys.exit(f"Could not locate an area named {queryarea}. " + 
                     'Please check your location query to make sure ' +
                     'it was entered correctly.')
        
        queryarea_printname = queryarea_name.split(",")[0]  
        
        url = 'http://overpass-api.de/api/interpreter'
        
        # Get the polygon boundary for the query area:
        query = f"""
        [out:json][timeout:5000];
        rel({queryarea_osmid});
        out geom;
        """
        
        r = requests.get(url, params={'data': query})
        
        datastruct = r.json()['elements'][0]
        if datastruct['tags']['type'] in ['boundary','multipolygon']:
            lss = []
            for coorddict in datastruct['members']:
                if coorddict['role']=='outer':
                    ls = []
                    for coord in coorddict['geometry']:
                        ls.append([coord['lon'],coord['lat']])
                    lss.append(LineString(ls))
        
            merged = linemerge([*lss])
            borders = unary_union(merged) # linestrings to a MultiLineString
            polygons = list(polygonize(borders)) 
            
            if len(polygons)==1:
                bpoly = polygons[0]
            else:
                bpoly = MultiPolygon(polygons)
        
        else:
            sys.exit(f"Could not retrieve the boundary for {queryarea}. " + 
                     'Please check your location query to make sure ' +
                     'it was entered correctly.')    
        if outfile:
            write_polygon2geojson(bpoly,outfile)
            
        return bpoly, queryarea_printname, queryarea_osmid
    
    def __bbox2poly(self,queryarea,outfile=False):
        
        # Parse the entered bounding box into a polygon:
        if len(queryarea)%2==0 and len(queryarea)!=0:                        
            if len(queryarea)==4:
                bpoly = box(*queryarea)
                queryarea_printname = (f"the bounding box: {list(queryarea)}")                        
            elif len(queryarea)>4:
                queryarea_printname = 'the bounding box: ['
                bpolycoords = []
                for i in range(int(len(queryarea)/2)):
                    bpolycoords = bpolycoords.append([queryarea[2*i], queryarea[2*i+1]])
                    queryarea_printname+= f'{queryarea[2*i]}, {queryarea[2*i+1]}, '
                bpoly = Polygon(bpolycoords)
                queryarea_printname = queryarea_printname[:-2]+']'
            else:
                raise ValueError('Less than two longitude/latitude pairs were entered to define the bounding box entry. ' + 
                                 'A bounding box can be defined by using at least two longitude/latitude pairs.') 
        else:
                raise ValueError('Incorrect number of elements detected in the tuple for the bounding box. ' 
                                 'Please check to see if you are missing a longitude or latitude value.')  
        if outfile:
            write_polygon2geojson(bpoly,outfile)
            
        return bpoly, queryarea_printname, None
    
    def get_boundary(self):

        queryarea = self.data
        
        if isinstance(queryarea,str):
            return self.__fetch_roi(queryarea)
        elif isinstance(queryarea,tuple):
            return self.__bbox2poly(queryarea) 

