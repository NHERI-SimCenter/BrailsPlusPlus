# Copyright (c) 2025 The Regents of the University of California
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 02-27-2025

"""
This module defines the class object for downloading ASCE Hazard data.

.. autosummary::

    ASCE_HAZARD_DATA_SCRAPER
"""

import concurrent.futures
import logging
from tqdm import tqdm
from shapely.geometry import box

from brails.types.asset_inventory import AssetInventory
from brails.utils import GeoTools
from brails.utils import ArcgisAPIServiceHelper

# Configure logging:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_ENDPOINT = ('https://gis.asce.org/arcgis/rest/services/ASCE722/'
                'w2022_Tile_RC_I/MapServer/2/query')


class ASCE_HAZARD_DATA_SCRAPER():
    """
    A class to generate footprint data using FEMA USA Structures building data.

    This class interacts with the FEMA USA Structures API to download
    building footprints, attributes (such as height), and additional metadata
    for a given geographic region. The class is built on top of the
    `FootprintScraper` class.

    Attributes:
        length_unit (str):
            Unit of length for building heights (default is 'ft').

    Methods:
        get_footprints(region: RegionBoundary):
            Obtains building footprints and creates an inventory for the
            specified region.
    """

    def __init__(self, input_dict: dict):
        """
        Initialize the class object.

        Args
            input_dict:
                A dictionary specifying length units; if "length" is not
                provided, "ft" is used as the default.
        """
        self.length_unit = input_dict.get('length', 'ft')

    def get_windspeeds(self, inventory: AssetInventory()) -> AssetInventory:
        """
        Retrieve building footprints and attributes for a specified region.

        This method divides the provided region into smaller cells, if
        necessary,  and then downloads building footprint data for each cell
        using the FEMA USA Structures API. The footprints and attributes are
        returned as an AssetInventory for buildings within the region.

        Args:
            region (RegionBoundary):
                The geographic region for which building footprints and
                attributes are to be obtained.

        Returns:
            AssetInventory:
                An inventory of buildings in the region, including their
                footprints and associated attributes (e.g., height).

        Raises:
            TypeError:
                If the 'region' argument is not an instance of the BRAILS++
                'RegionBoundary' class.

        Notes:
            - The region is split into smaller cells if the bounding area
              contains more than the maximum allowed number of elements per
              cell.
            - If the `plot_cells` flag is set to `True`, the cell boundaries
              are plotted and saved as an image.
            - The method creates a polygon mesh for the region, splits it if
              needed, and downloads building data for each cell in the region.
        """
        if not isinstance(inventory, AssetInventory()):
            raise TypeError("The 'inventory' argument must be an "
                            "'AssetInventory'")

        # Define a random bounding polygon. This specific API returns the same
        # output irrespective of the input geometry, yet does not allow
        # omitting the geometry argument:
        bpoly = box(-118.5534678, 33.9666583, 34.0505825, -118.4435336)

        # Get the number of elements allowed per cell by the API:
        api_tools = ArcgisAPIServiceHelper(API_ENDPOINT)

        datalist = api_tools.download_attr_from_api(bpoly, 'all')
