#!/usr/bin/env python
# coding: utf-8

# # Import BRAILS Importer

# In[1]:


# from brails.utils import Importer

import sys
sys.path.insert(0, "../../")
from brails.utils import Importer


# # Define Location Specific Parameters

# In[2]:


LOCATION_NAME = 'Fort Myers Beach, FL'
INVENTORY_OUTPUT = 'FortMyersInventory_HU.geojson'
NO_POSSIBLE_WORLDS = 1


# # Create and Importer object to Pull In Required BRAILS Modules

# In[3]:


importer = Importer()


# # Define the Region Object for the Region of Interest

# In[4]:


region_data = {"type": "locationName", "data": LOCATION_NAME}
region_boundary_class = importer.get_class("RegionBoundary")
region_boundary_object = region_boundary_class(region_data)


# # Get Raw NSI Data for the Defined Region

# In[5]:


nsi_class = importer.get_class('NSI_Parser')
nsi = nsi_class()
nsi_inventory = nsi.get_raw_data(region_boundary_object)


# # Get FEMA USA Footprints Data for the Defined Region

# In[6]:


scraper_class = importer.get_class('MS_FootprintScraper')
scraper = scraper_class({'length': 'ft'})
scraper_inventory = scraper.get_footprints(region_boundary_object)


# # Create a Baseline Inventory by Merging NSI Raw Data and USA Structures Footprints

# In[7]:


nsi_inventory = nsi.get_filtered_data_given_inventory(
    scraper_inventory, "ft", get_extended_features=True, add_features=[])


# # Fill Gaps in the Baseline Inventory by Imputing Missing Values

# In[8]:


knn_imputer_class = importer.get_class("KnnImputer")

imputer = knn_imputer_class(
    nsi_inventory, n_possible_worlds=NO_POSSIBLE_WORLDS,
    exclude_features=['lat', 'lon', 'fd_id'])
imputed_inventory = imputer.impute()


# # Predict Roof Type Information from Aerial Imagery

# In[9]:


google_satellite_class = importer.get_class('GoogleSatellite')
google_satellite = google_satellite_class()
images_aerial = google_satellite.get_images(imputed_inventory,
                                            'tmp/satellite/')

roof_shape_classifier_class = importer.get_class('RoofShapeClassifier')
roof_shape_classifier = roof_shape_classifier_class()
predictions = roof_shape_classifier.predict(images_aerial)


# # Getting Design Wind Speed, Flood Zone and Roof Type Information ,..., in Created Inventory

# In[10]:


for key, val in imputed_inventory.inventory.items():
    val.add_features({'DesignWindSpeed': 130,
                      'FloodZone': 'AE',
                      'RoofShape': predictions[key],
                      'AvgJanTemp':'Above',
                      'NumberOfUnits':1,
                      'LandCover': 'Open'})


# # Run Rulesets to Infer R2D-Required Data for Hazus Damage and Loss model

# In[12]:


imputed_inventory.change_feature_names({'erabuilt': 'YearBuilt',
                                        'constype': 'BuildingType',
                                        'numstories': 'NumberOfStories',
                                        'occupancy': 'OccupancyClass'})
hurricaneInferer = importer.get_class("HazusInfererWind")
inferer = hurricaneInferer(
    input_inventory=imputed_inventory, clean_features=True)
hazus_inferred_inventory = inferer.infer()


# # Validate the inventory and make correction if needed

# In[13]:


invalid_id, error_record = inferer.validate(hazus_inferred_inventory)


# In[14]:


inventory_corrrected = inferer.correct(hazus_inferred_inventory, invalid_id=invalid_id, weights={'StructureType':2})


# # Re-run Imputation to Fill Values That Cannot be Inferred by HAZUS Rulesets

# This is not recommended

# In[15]:


# imputer = knn_imputer_class(hazus_inferred_inventory, 
#                             n_possible_worlds=NO_POSSIBLE_WORLDS)
# hazus_inventory_final = imputer.impute()


# # Write the Created Inventory in a GeoJSON File

# In[17]:


_ = inventory_corrrected.write_to_geojson(
    output_file=INVENTORY_OUTPUT)


# In[ ]:




