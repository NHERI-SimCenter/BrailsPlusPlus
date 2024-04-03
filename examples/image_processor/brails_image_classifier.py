# Written: fmk 03/24
# Purpose: to test ImageSet, Importer and some satellite classifier

import sys
import inspect

# to add brails to the path .. so can play like brails installed
sys.path.insert(1,'../../')

from brails.utils.utils import Importer
from brails.types.image_set import ImageSet

importer = Importer()
arial_images = ImageSet();
street_images = ImageSet();
arial_images.set_directory("./images/satellite", True)
street_images.set_directory("./images/street", True)

#
# test importer and one of satellite classifiers
#

my_class = importer.get_class('RoofShapeClassifier')
my_classifier = my_class()
predictions = my_classifier.predict(arial_images)
print(predictions)

my_class = importer.get_class('RoofShapeLLM')
my_classifier = my_class()
predictions = my_classifier.predict(arial_images)
print(predictions)

my_class = importer.get_class('FacadeParser')
my_classifier = my_class()
# will need a new image set
predictions = my_classifier.predict(street_images) 
print(predictions)

