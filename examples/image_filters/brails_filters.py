# Copyright (c) 2024 The Regents of the University of California
#
# This file is part of BRAILS++.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Frank McKenna
# Barbaros Cetiner
#
# Last updated:
# 11-06-2024

"""
Example showing the use of BRAILS module that crops building images from panos.

 Purpose: Testing 1) get_class method of Importer
                  2) get_footprints method of USA_FootprintScraper module
                  3) get_images and print_info methods of GoogleSatellite and
                     GoogleStreetview
                  4) filter method of HouseView
                  5) dir_path method of an ImageSet object
"""

import os
from pathlib import Path
from brails import Importer

# This script needs a Google API Key to run.
# We suggest placing your API key in file apiKey.txt in the same directory as
# this script if you plan to commit changes to this example. This way, you do
# not risk accidentally uploading your API key (apiKey.txt is in .gitignore,
# so you have work to do to get it uploaded)

API_KEY_DIR = '../api_key.txt'
if os.path.exists(API_KEY_DIR):
    with open(API_KEY_DIR, 'r', encoding='utf-8') as file:
        api_key = file.readline().strip()  # Read first line & strip whitespace
else:
    raise FileNotFoundError('API key file not found. Please ensure the file'
                            f' exists at: {API_KEY_DIR}')


# Create the importer:
importer = Importer()

# Select a region and create its RegionBoundary:
region_data = {'type': 'locationName', 'data': 'Tiburon, CA'}
region_boundary_class = importer.get_class('RegionBoundary')
region_boundary_object = region_boundary_class(region_data)


# Get AssetInventory for buildings in the defined region via
# USA_FootprintScraper:
print('Trying USA_FootprintsScraper...')

usa_class = importer.get_class('USA_FootprintScraper')
usa_data = {'length': 'ft'}
instance2 = usa_class(usa_data)
usa_inventory = instance2.get_footprints(region_boundary_object)

print('Total number of assets detected using FEMA USA Structures:',
      len(usa_inventory.inventory))

# Subsample from the extracted assets to keep the image downloading step quick.
# Here, we are randomly sampling 10 buildings using a random seed value of 100:
small_inventory = usa_inventory.get_random_sample(10, 100)
print('\nTotal number of assets in the created subset:',
      len(small_inventory.inventory))


# Get street level imagery for the selected subset using GoogleStreetview:
google_input = {'apiKey': api_key}
google_street_class = importer.get_class('GoogleStreetview')
google_street = google_street_class(google_input)
images_street = google_street.get_images(small_inventory, 'tmp/street/')

images_street.print_info()

# Crop the obtained imagery such that they include just the buildings of
# interest:
filter_house = importer.get_class('HouseView')
filter_data = {}
image_filter = filter_house(filter_data)
filtered_image_set = image_filter.filter(images_street, 'filtered_images')
print('\nCropped images are available in ',
      Path(filtered_image_set.dir_path).resolve())
