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
# 10-01-2025

"""
This module defines a class for retrieving bridge data from NBI.

.. autosummary::

    NBIScraper
"""

from typing import Any, Dict, List
from shapely.geometry import Point, Polygon
from brails.constants import DEFAULT_UNITS
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


class NBIScraper:
    """
        A scraper for retrieving and processing National Bridge Inventory data.
    
        This class automates getting bridge data for a region from the NBI, 
        subdividing a region into smaller cells for efficient API calls, 
        filtering the results to include only bridges inside the specified
        region boundary, converting dimensional attributes into preferred 
        units, and organizing the data into an ``AssetInventory``.

        To import the :class:`NBIScraper`, use:
    
        .. code-block:: python
    
            from brails import Importer

            importer = Importer()
            nbi_scraper_class = importer.get_class('NBIScraper')
    
        Parameters:
            units (Dict[str, str]):
                A dictionary mapping measurement types (e.g., length, weight)
                to their preferred output units. Parsed from the optional
                ``input_dict`` argument or defaulted to predefined units.
            inventory (AssetInventory):
                An inventory object that stores all processed bridge assets for
                this scraper instance.
    """

    def __init__(self, input_dict: Dict[str, Any] = None):
        """
        Initialize an instance of the class with specified units.

        This constructor allows you to specify the length unit through the
        optional `input_dict`. If `input_dict` is not provided, the length
        unit defaults to 'ft' (feet). The inventory is also initialized when
        an instance is created.

        Args:
            input_dict (dict, optional):
                A dictionary that may contain keys 'length' and/or 'weight' 
                specifying the units to be in creating an ``AssetInventory`` 
                from NBI data. If provided, the values will be converted
                to lowercase and applied to the corresponding unit types. If 
                the dictionary is not provided or if a key is missing, default 
                units will be used: 'ft' (feet) for length and 'lb' (pounds) 
                for weight.
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
                A region boundary object representing the geographic region
                for which to retrieve bridge inventory data.

        Returns:
            AssetInventory:
                An inventory containing the bridge assets within the specified
                region.

        Raises:
            TypeError:
                If the ``region`` argument is not an instance of the
                ``RegionBoundary`` class. The error message will indicate that
                the ``region`` argument must be an instance of the
                ``RegionBoundary`` class.
        
        Example:
            >>> from brails.utils import Importer
            >>> importer = Importer()
            >>> # Define a small bounding box around Los Angeles area
            >>> region_data = {
            ...     'type': 'locationPolygon',
            ...     'data': (-118.278, 34.041, -118.271, 34.036)
            ... }
            >>> region_boundary = importer.get_class('RegionBoundary')
            >>> region = region_boundary(region_data)
            >>> nbi_scraper_class = importer.get_class('NBIScraper')
            >>> nbi_scraper = nbi_scraper_class()
            No length unit specified. Using default: 'ft'.
            No weight unit specified. Using default: 'lb'.
            >>> inventory = nbi_scraper.get_assets(region)
            Meshing the defined area...
            Meshing complete. Covered the bounding box: (-118.278, 34.041, 
            -118.271, 34.036) with a single rectangular cell.
            Obtaining the bridge attributes for each cell: 100%|██████████| 1/1
            [00:00<00:00,  4.89it/s]
            Found a total of 4 bridges.
            >>> _ = inventory.write_to_geojson('nbi_scraper_test.geojson')
            Wrote 4 assets to /home/bacetiner/nbi_scraper_test.geojson
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
        boundary_polygon, _, _ = region.get_boundary(print_progress=False)

        # Identify the cells that are inside the bounding polygon and record
        # their data:
        data, data_to_filter = [], []
        for cell in final_cells:
            if boundary_polygon.contains(cell):
                data.extend(results.get(cell, []))
            else:
                data_to_filter.extend(results.get(cell, []))

        # Filter the data within the cells that are not contained in the
        # bounding polygon such that only the points within the bounding
        # polygon are retained:
        for item in data_to_filter:
            if boundary_polygon.contains(Point(item['geometry']['x'],
                                               item['geometry']['y'])):
                data.append(item)

        # Display the number of elements detected:
        element_number = len(data)
        element = 'bridge' if element_number==1 else 'bridges'
        print(f'\nFound a total of {element_number} {element}.')


        # Save the results in the inventory:
        for index, item in enumerate(data):
            geometry = [[item['geometry']['x'], item['geometry']['y']]]
            asset_features = {**item['attributes'], 'type': ASSET_TYPE}
            
            for feature, feature_unit in DIMENSIONAL_ATTR.items():
                feature_value = asset_features.get(feature)
                
                # Attempt to convert the feature value to float; set to None if 
                # conversion fails:
                try:
                    feature_value = float(feature_value)
                except (ValueError, TypeError):
                    feature_value = None
                
                if feature_value is not None:
                    # Determine the type of unit and set the target conversion
                    # unit: 
                    unit_type = UnitConverter.get_unit_type(feature_unit)
                    target_unit = self.units[unit_type]
                    
                    # Convert the feature value to the target unit and store 
                    # the converted value back in the asset:
                    asset_features[feature] = UnitConverter.convert_unit(
                        feature_value,
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
