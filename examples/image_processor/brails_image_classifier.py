'''
Purpose: Testing ImageSet, Importer and an aerial image classifier
'''

# Written: fmk 04/24
# Copyright BSD2

import sys
import importlib.util

#if importlib.util.find_spec("brails") is None:
sys.path.insert(1,'../../')

from brails.types.image_set import ImageSet    
from brails.utils.utils import Importer

importer = Importer()
aerial_images = ImageSet();
street_images = ImageSet();
aerial_images.set_directory("./images/satellite_easy", True)
street_images.set_directory("./images/street", True)

aerial_images.print()

# Test importer and one of the aerial imagery classifiers, RoofShapeClassifier:
print('ROOF_SHAPE_CLASSICAL PREDICTIONS')
my_class = importer.get_class('RoofShapeClassifier')
my_classifier = my_class()
predictions = my_classifier.predict(aerial_images)
print(predictions)

print('ROOF_SHAPE_VLM PREDICTIONS')
my_class = importer.get_class('RoofShapeVLM')
my_classifier = my_class()
predictions = my_classifier.predict(aerial_images)
print(predictions)

print('NFLOORS_VLM PREDICTIONS')
my_class = importer.get_class('NFloorVLM')
my_classifier = my_class()
predictions = my_classifier.predict(street_images)
print(predictions)

# Test importer and one of the street-level imagery classifiers, 
# OccupancyClassifier:
print('OCCUPANCY_CLASS_CLASSICAL PREDICTIONS')
my_class = importer.get_class('OccupancyClassifier')
my_classifier = my_class()
predictions = my_classifier.predict(street_images) 
print(predictions)

my_class = importer.get_class('VLMSegmenter')
my_segmenter = my_class()
my_segmenter.predict(street_images)