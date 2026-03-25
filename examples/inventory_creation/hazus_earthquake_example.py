#!/usr/bin/env python
# coding: utf-8

#
# Purpose to create the building inventory of a region of a region of interest
# for R2D to study the effect of an earthquake event  on the region.
#

# This is a script created from the notebook of same name.
# Original notebook created by: Adam Zsarnoczay early 2026


from brails.types import RegionBoundary

region_boundary_object = RegionBoundary(
    {
        "type": "locationName",
        "data": "Berkeley, CA"
    }
)

geometry, description, osm_id = region_boundary_object.get_boundary()


# In[70]:


from brails.scrapers import OvertureMapsFootprintScraper

bldg_inventory = OvertureMapsFootprintScraper({'length': 'ft'}).get_footprints(region_boundary_object)

# only use the footprint geometry and remove all other metadata for clarity
bldg_inventory.remove_features(bldg_inventory.get_all_asset_features())


# In[71]:


from brails.scrapers import NSI_Parser

nsi_points = NSI_Parser().get_raw_data(region_boundary_object)

# Rename the features we need later using standard SimCenter labels
feature_rename_map = {
    'fd_id': 'fd_id',
    'type': 'type',
    'bldgtype': 'BuildingType',
#    'found_type': 'FoundationType',
    'found_ht': 'FirstFloorElevation',
    'pop2amu65': 'NightPopulationUnder65',
    'pop2amo65': 'NightPopulationOver65',
    'pop2pmu65': 'DayPopulationUnder65',
    'pop2pmo65': 'DayPopulationOver65',
    'x': 'Longitude',
    'y': 'Latitude',
    'sqft': 'PlanArea',
    'num_story': 'NumberOfStories',
    'students': 'StudentPopulation',
    'med_yr_blt': 'YearBuilt',
    'occtype': 'OccupancyClass'
}

nsi_points.change_feature_names(feature_rename_map)

# remove all other features for clarity
features_to_remove = nsi_points.get_all_asset_features()
features_to_remove.difference_update(feature_rename_map.values())
nsi_points.remove_features(features_to_remove)


# In[72]:


from brails.aggregators import BasicPointsToPolygonsAllocator

BasicPointsToPolygonsAllocator(
    polygon_inventory=bldg_inventory,
    point_inventory=nsi_points,
).allocate(
    use_convex_hull = True,
    buffer_dist = 10.0
)


# In[73]:


from brails.types import AssetInventory
from copy import deepcopy
import pandas as pd
import random

from brails.validators.hazusEarthquake.hazus_earthquake_validator import HazusEarthquakeValidator
validator = HazusEarthquakeValidator()
bldg_inventory_fixed, issues_df = validator.fix_inventory(bldg_inventory);

issues_df

# In[75]:


from brails.imputers import KnnImputer

bldg_inventory_imputed = KnnImputer(
    bldg_inventory_fixed,
    n_possible_worlds=1,
    exclude_features=['Latitude','Longitude','fd_id','id']
).impute()


# In[76]:


bldg_inventory_imputed_fixed, issues_df = validator.fix_inventory(bldg_inventory_imputed)

issues_df


# In[77]:


from brails.inferers import HazusInfererEarthquake

# add features shared by all buildings
for asset_id, asset in bldg_inventory_imputed_fixed.inventory.items():
    asset.add_features({
        'Income': 78672,     # This could be based on Households assigned to this building
        'SplitLevel': "No",  # This could be smarter since we know split level from NSI
    })

bldg_inventory_final = HazusInfererEarthquake(
    input_inventory=bldg_inventory_imputed_fixed,
    clean_features=False
).infer()


# In[78]:


for asset_id, asset in bldg_inventory_final.inventory.items():

    # we have to adjust the RES3 OccupancyClasses
    occupancy = asset.features.get('OccupancyClass', None)

    # map occupancy classes to the strict Hazus names
    if occupancy[:4] == 'RES3':
        asset.add_features({
            'OccupancyClass': occupancy[:4]
        })

# In[45]:


# Buildings
bldg_inventory_final.write_to_geojson('bldg_inventory_EQ.geojson')


# In[79]:


# to get a CSV
import geopandas as gpd

bldg_geojson = bldg_inventory_final.get_geojson()
bldg_gdf = gpd.GeoDataFrame.from_features(
    bldg_geojson['features'],
    crs = 'EPSG:4326'
)

centroids = bldg_gdf.geometry.centroid
bldg_gdf['Longitude'] = centroids.x
bldg_gdf['Latitude'] = centroids.y

bldg_df = bldg_gdf.drop(columns=['geometry'])

bldg_df.to_csv('bldg_inventory_EQ.csv')

