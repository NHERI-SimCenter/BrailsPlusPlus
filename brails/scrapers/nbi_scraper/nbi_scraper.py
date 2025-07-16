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
# 07-16-2025

"""
This module defines a class for retrieving bridge data from NBI.

.. autosummary::

    NBIScraper
"""

from typing import Any, Dict, List
from shapely.geometry import Point, Polygon
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils import ArcgisAPIServiceHelper, UnitConverter

# Type of asset covered by the dataset:
ASSET_TYPE = 'bridge'

# API endpoint to access National Bridge Inventory data:
API_ENDPOINT = (
    'https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/'
    'NTAD_National_Bridge_Inventory/FeatureServer/0/query'
)

# Attributes with associated units and the units they are defined in within
# the dataset:
DIMENSIONAL_ATTR = {'MIN_VERT_CLR_010': 'm',
                    'KILOPOINT_011': 'km',
                    'DETOUR_KILOS_019': 'km',
                    'APPR_WIDTH_MT_032': 'm',
                    'NAV_VERT_CLR_MT_039': 'm',
                    'NAV_HORR_CLR_MT_040': 'm',
                    'HORR_CLR_MT_047': 'm',
                    'MAX_SPAN_LEN_MT_048': 'm',
                    'STRUCTURE_LEN_MT_049': 'm',
                    'LEFT_CURB_MT_050A': 'm',
                    'RIGHT_CURB_MT_050B': 'm',
                    'ROADWAY_WIDTH_MT_051': 'm',
                    'DECK_WIDTH_MT_052': 'm',
                    'VERT_CLR_OVER_MT_053': 'm',
                    'VERT_CLR_UND_054B': 'm',
                    'LAT_UND_MT_055B': 'm',
                    'LEFT_LAT_UND_MT_056': 'm',
                    'INVENTORY_RATING_066': 'ton',
                    'IMP_LEN_MT_076': 'm',
                    'MIN_NAV_CLR_MT_116': 'm'}

# Default units to apply when not explicitly provided by the user:
DEFAULT_UNITS = {'length': 'ft',
                 'weight': 'lb'}


class NBIScraper:
    """
    A class for scraping and processing National Bridge Inventory (NBI) data.

    The class handles tasks such as fetching and processing bridge data,
    meshing the region into smaller cells for efficient data retrieval, and
    organizing the results into an AssetInventory.

    Attributes:
        length_unit (str):
            The unit of length (default is 'ft').
        inventory (AssetInventory):
            An inventory object that holds the assets for this instance.

    Methods:
        get_assets(region: RegionBoundary) -> AssetInventory:
            Retrieves bridge inventory data within a given region boundary,
            processes the data, and returns the inventory of assets.
    """

    def __init__(self, input_dict: Dict[str, Any] = None):
        """
        Initialize an instance of the class with a specified length unit.

        This constructor allows you to specify the length unit through the
        optional `input_dict`. If `input_dict` is not provided, the length
        unit defaults to 'ft' (feet). The inventory is also initialized when
        an instance is created.

        Args:
            input_dict (dict, optional):
                A dictionary that may contain a key 'length' specifying the
                length unit to be used. If 'length' is provided in the
                dictionary, it will be used to set the length unit (the value
                will be converted to lowercase). If the dictionary is not
                provided or if 'length' is not specified, 'ft' (feet) will be
                used as the default length unit.
        """
        # Parse units from input_dict or fall back to default units:
        self.units = UnitConverter.parse_units(input_dict or {}, DEFAULT_UNITS)
        self.inventory = AssetInventory()

    def get_assets(self, region: RegionBoundary) -> AssetInventory:
        """
        Retrieve bridge inventory within a given region.

        This method processes the region boundary, splits it into smaller
        cells, fetches the necessary bridge data for each cell, and returns the
        processed bridge asset inventory.

        Args:
            region (RegionBoundary):
                A `RegionBoundary` object representing the geographic region
                for which to retrieve bridge inventory data.

        Returns:
            AssetInventory:
                An `AssetInventory` object containing the bridge assets within
                the specified region.

        Raises:
            TypeError:
                If the `region` argument is not an instance of the
                `RegionBoundary` class. The error message will indicate that
                the 'region' argument must be an instance of the
                `RegionBoundary` class.
        """
        if not isinstance(region, RegionBoundary):
            raise TypeError("The 'region' argument must be an instance of the "
                            "'RegionBoundary' class.")

        # Download bridge data for each cell:
        api_tools = ArcgisAPIServiceHelper(API_ENDPOINT)
        results, final_cells = api_tools.download_all_attr_for_region(
            region,
            task_description='Obtaining the bridge attributes for each cell'
        )

        # Process the downloaded data and save it in an AssetInventory:
        self._process_data(region, final_cells, results)
        return self.inventory

    def _process_data(
        self,
        region: RegionBoundary,
        final_cells: List[Polygon],
        results: Dict[Polygon, List[Dict[str, Any]]]
    ) -> None:
        """
        Process the downloaded NBI data and store it in an AssetInventory.

        This method filters bridge data retrieved from each cell based on
        whether the geometry lies within the region boundary, converts units if
        needed, and constructs asset objects for the inventory.

        Args:
            region (RegionBoundary):
                The region object containing the boundary polygon used to
                filter relevant bridge data.
            final_cells (List[Polygon]):
                List of polygonal cells that subdivide the region and contain
                bridge data.
            results (Dict[Polygon, List[Dict[str, Any]]]):
                A mapping from each cell polygon to a list of dictionaries,
                each representing a bridge and its associated attributes.
        """
        # Obtain the boundary polygon for the region:
        boundary_polygon, _, _ = region.get_boundary()

        # Identify the cells that are inside the bounding polygon and record
        # their data:
        data, data_to_filter = [], []
        for cell in final_cells:
            if boundary_polygon.contains(cell):
                data.extend(results.get(cell, []))
            else:
                data_to_filter.extend(results.get(cell, []))

        # Filter the data within the cells that are not contained in the
        # bounding polygon such that only the points withing the bounding
        # polygon are retained:
        for item in data_to_filter:
            if boundary_polygon.contains(Point(item['geometry']['x'],
                                               item['geometry']['y'])):
                data.append(item)

        # Display the number of elements detected:
        print(f'\nFound a total of {len(data)} bridges.')

        # Save the results in the inventory:
        for index, item in enumerate(data):
            geometry = [[item['geometry']['x'], item['geometry']['y']]]
            asset_features = {**item['attributes'], 'type': ASSET_TYPE}
            for feature, feature_unit in DIMENSIONAL_ATTR.items():
                feature_Value = asset_features.get(feature)
                if feature_Value is not None:
                    unit_type = UnitConverter.get_unit_type(feature_unit)
                    target_unit = self.units[unit_type]
                    asset_features[feature] = UnitConverter.convert_unit(
                        feature_Value,
                        feature_unit,
                        target_unit
                    )

            # Create the Asset and add it to the inventory:
            asset = Asset(index, geometry, asset_features)
            self.inventory.add_asset(index, asset)

        # TODO: Remove duplicate bridges resulting from the way NBI stores data
        # across different states. Solution likely requires using
        # structure_number_008, othr_state_struc_no_099, and
        # other_state_pcnt_098b fields.
