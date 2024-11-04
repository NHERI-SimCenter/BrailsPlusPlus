"""Purpose: Testing ImageSet, Importer and an aerial image classifier."""

# Written: fmk 04/24
# Modified: Barbaros Cetiner 10/24
# Copyright BSD2

from brails.types.image_set import ImageSet
from brails.utils import Importer

importer = Importer()
aerial_images = ImageSet()
aerial_images.set_directory("./images/satellite_easy", True)
street_images = ImageSet()
street_images.set_directory("./images/street", True)

# Test importer using a couple of the aerial imagery classifiers:
aerial_images.print_info()
print('ROOF SHAPE PREDICTIONS USING MODEL TRAINED ON CUSTOM DATASET:')
my_class = importer.get_class('RoofShapeClassifier')
my_classifier = my_class()
predictions = my_classifier.predict(aerial_images)
print(predictions)

print('\nROOF SHAPE PREDICTIONS USING CLIP VLM:')
my_class = importer.get_class('RoofShapeVLM')
my_classifier = my_class()
predictions = my_classifier.predict(aerial_images)
print(predictions)

# Test importer and a couple of the street-level imagery classifiers:
street_images.print_info()
print('\nOCCUPANCY CLASS PREDICTIONS USING MODEL TRAINED ON CUSTOM DATASET:')
my_class = importer.get_class('OccupancyClassifier')
my_classifier = my_class()
predictions = my_classifier.predict(street_images)
print(predictions)

print('\nNUMBER OF FLOOR PREDICTIONS USING CLIP VLM:')
my_class = importer.get_class('NFloorVLM')
my_classifier = my_class()
predictions = my_classifier.predict(street_images)
print(predictions)
