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
# 08-01-2025

"""
This module defines abstract FootprintScraper class.

.. autosummary::

    FootprintScraper
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from numpy import append, arctan2, cos, deg2rad, diff, pi, sin, sqrt
from shapely.geometry import Polygon

from brails.types.asset_inventory import Asset, AssetInventory
from brails.types.region_boundary import RegionBoundary
from brails.utils import UnitConverter

# Global constants for plan area calculations and asset typing:
EARTH_RADIUS_FT = 20925721.784777  # Earth's radius in feet (WGS-84)
ASSET_TYPE = 'building'


class FootprintScraper(ABC):
    """
    Abstract base class for getting building footprints within a region.

    This class defines the interface for any footprint scraper implementation.

    Attributes:
        name (str):
            Name or identifier for the scraper.
        footprints (List):
            A list to store retrieved footprint geometries.
        centroids (List):
            A list to store centroid coordinates of the footprints.

    Methods:
        get_footprints(region):
            Abstract method to return building footprints for a given region.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the footprint scraper.

        Args:
            name (str):
                Name or identifier for the scraper.
        """
        self.name: str = name
        self.footprints = []
        self.centroids = []

    @abstractmethod
    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        Retrieve building footprints and attributes within a specified region.

        This method must be implemented by subclasses to provide access to
        building-level footprint and attribute data for a given geographic
        boundary. It is typically used by downstream components to extract
        spatial information for analysis or visualization.

        Args:
            region (RegionBoundary):
                The geographic region of interest.

        Returns:
            AssetInventory:
                An inventory containing the footprints of buildings within the
                specified region.
        """
        pass

    def _polygon_area(
        self,
        lats: Union[List[float], Tuple[float, ...]],
        lons: Union[List[float], Tuple[float, ...]],
        length_unit: str = 'ft'
    ) -> float:
        """
        Calculate the approximate area of a polygon on the Earth's surface.

        The area is computed using Green's Theorem applied to a spherical
        surface, which approximates the Earth as a perfect sphere.

        Args:
            lats (list or tuple of float):
                Latitudes of the polygon vertices in degrees.
            lons (list or tuple of float):
                Longitudes of the polygon vertices in degrees.
            length_unit (str):
                Desired output length unit to use for area calculation
                (e.g., 'ft', 'm'). Defaults to 'ft'.

        Returns:
            float:
                Area of the polygon in the requested unit.
        """
        lats = deg2rad(lats)
        lons = deg2rad(lons)

        # Close the polygon if not already closed
        if lats[0] != lats[-1] or lons[0] != lons[-1]:
            lats = append(lats, lats[0])
            lons = append(lons, lons[0])

        # Compute spherical coordinates relative to origin (0,0):
        a = sin(lats / 2) ** 2 + cos(lats) * sin(lons / 2) ** 2
        colat = 2 * arctan2(sqrt(a), sqrt(1 - a))

        # Azimuths relative to (0,0):
        az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2 * pi)

        # Angle differences, adjusted to [-pi, pi]:
        daz = diff(az)
        daz = (daz + pi) % (2 * pi) - pi

        # Midpoint colatitudes:
        colat_mid = colat[:-1] + diff(colat) / 2

        # Green's theorem integral:
        integrands = (1 - cos(colat_mid)) * daz
        spherical_area_ratio = abs(sum(integrands)) / (4 * pi)
        spherical_area_ratio = min(
            spherical_area_ratio, 1 - spherical_area_ratio
        )

        # Convert to area in square feet:
        area_sqft = spherical_area_ratio * 4 * pi * EARTH_RADIUS_FT**2

        # Convert to requested unit:
        return UnitConverter.convert_unit(area_sqft, 'ft2', f'{length_unit}2')

    def _create_asset_inventory(
        self,
        footprints: list,
        attributes: dict,
        length_unit: str
    ) -> AssetInventory:
        """
        Construct AssetInventory object from footprints and attributes.

        This method is intended to be used by subclasses to assemble an
        AssetInventory from raw geometric and attribute data.

        Args:
            footprints (list):
                A list of building footprints. Each footprint is a list of
                [lon, lat] coordinate pairs defining the polygon.
            attributes (dict):
                A dictionary of attribute lists, where each key corresponds to
                an attribute name (e.g., "height", "material") and each value
                is a list with one entry per building.
            length_unit (str):
                Unit of length to use for derived measurements
                (e.g., 'ft', 'm').

        Returns:
            AssetInventory:
                An inventory containing building assets for the specified
                region.
        """
        # TODO: Need to calculate footprint areas after removing bad polygons.
        # Compute footprint areas:
        attributes['footprintArea'] = []
        for footprint in footprints:
            lons, lats = zip(*footprint)
            area = int(self._polygon_area(list(lats), list(lons), length_unit))
            attributes['footprintArea'].append(area)

        # Calculate centroids of the footprints and remove footprint data that
        # does not form a polygon:
        self.footprints = []
        self.centroids = []
        valid_indices = []

        for index, footprint in enumerate(footprints):
            try:
                polygon = Polygon(footprint)
                centroid = polygon.centroid
                self.footprints.append(footprint)
                self.centroids.append(centroid)
                valid_indices.append(index)
            except Exception as e:
                print(
                    f'[Warning] Invalid footprint at index {index}, '
                    f'skipping. Error: {e}'
                )
                continue

        # Filter attributes to keep only valid entries:
        for key in attributes:
            attributes[key] = [attributes[key][i] for i in valid_indices]

        # Assemble AssetInventory:
        inventory = AssetInventory()
        for index, footprint in enumerate(self.footprints):
            asset_features = {'type': ASSET_TYPE}
            for key, values in attributes.items():
                value = values[index]
                asset_features[key] = "NA" if value is None else value
            # TODO: Need remove NA's from the inventory. # Note: This may
            # affect imputers. Need to verify compatibility before proceeding.

            asset = Asset(index, footprint, asset_features)
            inventory.add_asset(index, asset)

        return inventory
