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
Example demonstrating the use of NSI scraper module .

 Purpose: Testing 1) get_class method of Importer
                  2) get_raw_data_given_boundary and
                     get_filtered_data_given_inventory methods of NSI_Parser
                  3) get_footprints method of OSM_FootprintScraper module
                  4) print_info and write_to_geojson method of AssetInventory
                     objects
"""

import os
import json
from brails import Importer

# Create an importer to get the required classes:
importer = Importer()

# Select a region and create its RegionBoundary:
region_data = {'type': 'locationName', 'data': 'Tiburon, CA'}
region_boundary_class = importer.get_class('RegionBoundary')
region_boundary_object = region_boundary_class(region_data)

# Use NSI_Parser to get the NSI raw data for the specified region:
nsi_class = importer.get_class('NSI_Parser')
nsi = nsi_class()
inventory = nsi.get_raw_data_given_boundary(region_boundary_object, 'ft')
inventory.print_info()

# Get the footprints for the defined region from OSM:
print('Trying OSM_FootprintsScraper...')

osm_class = importer.get_class('OSM_FootprintScraper')
osm_data = {'length': 'ft'}
osm = osm_class(osm_data)
osm_inventory = osm.get_footprints(region_boundary_object)
print('Total number of assets detected using OSM:',
      len(osm_inventory.inventory))

# Get a small subset of the obtained footprints:
osm_inventory_subset = osm_inventory.get_random_sample(5, 200)
print('\nTotal number of assets in the randomly selected subset:',
      len(osm_inventory_subset.inventory))

print('\n******** SMALL INVENTORY *********')
osm_inventory_subset.print_info()

# Get the NSI raw data for this subset:
print('\n******** NSI RAW DATA FOR SMALL INVENTORY *********')
subset_nsi_raw_data = nsi.get_raw_data_given_inventory(
    osm_inventory_subset, 'ft')
subset_nsi_raw_data.print_info()

# Get the NSI filtered data for this subset:
print('\n******** NSI FILTERED DATA FOR SMALL INVENTORY *********')
subset_nsi_processed_data = nsi.get_filtered_data_given_inventory(
    osm_inventory_subset, 'ft')
subset_nsi_processed_data.print_info()

# Write the extracted inventory to a GeoJSON file:
FILE_PATH = 'tmp/smallinv.geojson'
directory = os.path.dirname(FILE_PATH)
os.makedirs(directory, exist_ok=True)

geojson = subset_nsi_processed_data.write_to_geojson()
with open(FILE_PATH, 'w', encoding='utf8') as f:
    json.dump(geojson, f, indent=2)
