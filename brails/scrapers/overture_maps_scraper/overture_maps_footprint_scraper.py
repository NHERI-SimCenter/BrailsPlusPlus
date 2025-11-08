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
# 08-14-2025

"""
This module define the concrete class for scraping OvertureMaps footprint data.

.. autosummary::

    OvertureMapsFootprintScraper
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from shapely import wkb

from brails.constants import DEFAULT_UNITS
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils import GeoTools, UnitConverter
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import \
    OvertureMapsScraper

# Define global variables:
OVERTURE_TYPES = ['building', 'building_part']
ASSET_TYPE = 'building'

# Attributes with associated units and the units they are defined in within
# the dataset:
DIMENSIONAL_ATTR = {'height': 'm',
                    'min_height': 'm',
                    'roof_height': 'm'}


class OvertureMapsFootprintScraper(OvertureMapsScraper):
    """
    Scraper for extracting Overture Maps buildings and building-parts.

    This class extends
    :class:`~brails.scrapers.overture_maps_scraper.overture_maps_scraper.OvertureMapsScraper`
    and provides methods to load and filter Overture asset
    datasets based on a region boundary. Assets are stored in an
    :class:`~brails.types.asset_inventory.AssetInventory` and can be exported
    as GeoJSON.

    Attributes:
        units (str):
            Length unit used for spatial calculations. Parsed from
            ``input_dict`` or defaults to ft.
        inventory (AssetInventory):
            Stores the assets retrieved by the scraper.
        overture_release (str):
            The specific Overture Maps release being used. Falls back to the
            latest release if not specified in ``input_dict``.
    """

    def __init__(self, input_dict: Dict[str, Any] = None):
        """
        Initialize the class with user-defined or default settings.

        This constructor sets the unit system, initializes an empty asset
        inventory, and determines which Overture Maps release to use.
        Configuration options can be provided via the optional ``input_dict``.

        Args:
            input_dict (dict, optional):
                A configuration dictionary that may include:

                - 'length' (str): Length unit to use (e.g., 'ft', 'm').
                  Defaults to the unit defined in ``DEFAULT_UNITS`` if not
                  provided.
                - 'overtureRelease' (str): Specific Overture Maps release name
                  to use. If omitted, the latest available release from
                  meth:`~brails.scrapers.overture_maps_scraper.overture_maps__scraper.OvertureMapsScraper.fetch_release_names` 
                  will be selected, with a warning.

        Notes:
            - Units are parsed and validated through
              :meth:`~brails.utils.unit_converter.UnitConverter.parse_units`.
            - An empty :class:AssetInventory` is created upon initialization.
            - If ``input_dict`` is not provided, all defaults will be used.
        """
        # Parse units from input_dict or fall back to default units:
        self.units = UnitConverter.parse_units(input_dict or {}, DEFAULT_UNITS)
        self.inventory = AssetInventory()

        # Parse Overture Maps release from input dict or fall back to latest:
        overture_maps_releases = self.fetch_release_names()
        self.overture_release = input_dict.get('overtureRelease')
        if not self.overture_release:
            print(
                "'overtureRelease' not defined in input. Falling back to "
                f'latest release: {overture_maps_releases[0]}'
            )
            self.overture_release = overture_maps_releases[0]

    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        Retrieve Overture building assets within a specified region.

        This method loads Overture dataset for assets of 'building' and
        'building_part' types that are within the boundary of the given
        region. Resulting assets are save as an
        :class:`~brails.types.asset_inventory.AssetInventory`.

        Args:
            region (RegionBoundary)
                A BRAILS++ region boundary object defining the coordinate
                information for the region of interest.

        Returns:
            AssetInventory: An inventory of assets within the specified region.

        Notes:
            - Only assets that intersect the provided region are included.
            - Columns named "names" are dropped to avoid GeoJSON export issues.
            - Asset attributes exclude ``None`` and ``NaN`` values.
            - The inventory is stored in ``self.inventory`` and also returned.

        Example:
            First import the BRAILS++ :class:`~brails.utils.importer.Importer`
            class.

            >>> from brails import Importer
            >>> importer = Importer()

            Create a region boundary object for Berkeley, CA.

            >>> region_data = {"type": "locationName", "data": 'Berkeley, CA'}
            >>> region_boundary_class = importer.get_class('RegionBoundary')
            >>> region_boundary_object = region_boundary_class(region_data)

            Get building inventory data using the '2024-07-22.0' release of
            Overture Maps for Berkeley, CA.

            >>> scraper_class = importer.get_class(
            ... 'OvertureMapsFootprintScraper'
            ... )
            >>> scraper_object = scraper_class(
            ...     input_dict={'overtureRelease': '2024-07-22.0'}
            ... )
            No length unit specified. Using default: 'ft'.
            No weight unit specified. Using default: 'lb'.
            >>> inventory = scraper_object.get_footprints(
            ...     region_boundary_object
            ... )
            Searching for Berkeley, CA...
            Found Berkeley, Alameda County, California, United States
            Reading dataset batches: 339it [00:39,  8.49it/s]
            Reading dataset batches: 12it [00:02,  4.99it/s]
            Finding the assets within the specified area...
            Found a total of 36181 assets within the specified area.

            Write the obtained inventory in ``berkeley_buildings.geojson``.

            >>> _ = inventory.write_to_geojson('berkeley_buildings.geojson')
            Wrote 36181 assets to /home/bacetiner/Documents/SoftwareTesting/
            berkeley_buildings.geojson
        """
        bpoly, _,  _ = region.get_boundary()
        bbox = bpoly.bounds

        dfs = []
        # Load data for each overture type:
        for overture_type in OVERTURE_TYPES:
            df = self._read_to_pandas(
                self.overture_release,
                overture_type,
                bbox=bbox
            )
            dfs.append(df)

        dataset = pd.concat(dfs, ignore_index=True)

        # Convert WKB to Shapely geometries:
        dataset["geometry"] = dataset["geometry"].apply(
            lambda wkb_bytes: wkb.loads(wkb_bytes))

        print('\nFinding the assets within the specified area...\n')

        # Filter rows that intersect with the bounding polygon:
        intersects_mask = [geom.intersects(bpoly)
                           for geom in dataset["geometry"]]
        intersecting_rows = dataset[intersects_mask].copy()

        # Format sources
        intersecting_rows['sources'] = intersecting_rows['sources'].apply(
            self._format_sources)

        # TODO: Figure out how to parse names
        # Drop problematic columns for GeoJSON export:
        intersecting_rows.drop(columns=["names"], inplace=True)

        # Convert geometry to list of coordinates:
        intersecting_rows["geometry_coords"] = \
            intersecting_rows["geometry"].apply(
                GeoTools.geometry_to_list_of_lists
        )
        intersecting_rows["overture_id"] = intersecting_rows["id"]
        intersecting_rows.reset_index(drop=True, inplace=True)

        # Build inventory:
        for index, row in intersecting_rows.iterrows():
            geometry = row["geometry_coords"]

            # Filter attributes:
            attributes = {
                k: v for k, v in row.items()
                if k not in ("geometry", "geometry_coords", 'bbox', 'id')
                and v is not None
                and not (isinstance(v, float) and np.isnan(v))
            }

            # Convert dimensional attributes:
            for attr, attr_unit in DIMENSIONAL_ATTR.items():
                unit_type = UnitConverter.get_unit_type(attr_unit)
                target_unit = self.units[unit_type]
                if attr in attributes:
                    try:
                        attributes[attr] = UnitConverter.convert_length(
                            attributes[attr],
                            attr_unit,
                            target_unit)
                    except Exception as e:
                        print(
                            f"Warning: could not convert {attr} for index "
                            f"{index}: {e}"
                        )

            # Merge with asset type
            asset_features = {**attributes, "type": ASSET_TYPE}

            asset = Asset(index, geometry, asset_features)
            self.inventory.add_asset(index, asset)
        print(
            f'Found a total of {len(intersecting_rows)} assets within the '
            'specified area.'
        )
        return self.inventory
