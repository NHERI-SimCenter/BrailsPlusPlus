"""Purpose: Testing ImageSet, Importer and an street-level detector."""

# Written: Barbaros Cetiner 09/24
# Copyright BSD2

from brails.types.image_set import ImageSet
from brails.utils.utils import Importer

importer = Importer()
street_images = ImageSet()
street_images.set_directory("./images/street", True)

street_images.print_info()

# Test importer and one of the aerial imagery classifiers, RoofShapeClassifier:
print('Number of floors predictions using a detector model trained on a custom'
      ' dataset:')
my_class = importer.get_class('NFloorDetector')
my_classifier = my_class()
predictions = my_classifier.predict(street_images)
print(predictions)

print('Chimney detection using a detector model trained on a custom'
      ' dataset:')
my_class = importer.get_class('ChimneyDetector')
my_classifier = my_class()
predictions = my_classifier.predict(street_images)
print(predictions)

print('Garage detection using a detector model trained on a custom'
      ' dataset:')
my_class = importer.get_class('GarageDetector')
my_classifier = my_class()
predictions = my_classifier.predict(street_images)
print(predictions)
