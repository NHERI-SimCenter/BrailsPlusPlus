# Writtten:
#    fmk (abstract class)  03/24
#    Barbaros Cetiner (subclass needed methods: _polygon_area and create_asset_inventory)
# Licensce: BSD-2

"""
This module defines abstract FootprintScraper class

.. autosummary::

    FootprintScraper
"""

from abc import ABC, abstractmethod
from shapely.geometry import Polygon

from brails.types.asset_inventory import AssetInventory
from brails.types.region_boundary import RegionBoundary

from numpy import arctan2, cos, sin, sqrt, pi, append, diff, deg2rad




class FootprintScraper(ABC):
    """
    Abstract base class representing a class that optains footprints for a region.

      Methods:
         get_footprints(location): An abstract method to return the footprint given a location
    """

    def __init__(self, name):
        self.name = name
        self.footprints = []
        self.centroids = []

    @abstractmethod
    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        An abstract class that must be implemented by subclasses.

        This method will be used by the caller to obtain the footprints for builings in an area.

        Args:
              region (Region):
                   The region of interest.

        Returns:
              BuildingInventory:
                    A building inventory for buildings in the region.

        """
        pass

    def _polygon_area(self, lats, lons, length_unit):

        radius = 20925721.784777  # Earth's radius in feet

        lats = deg2rad(lats)
        lons = deg2rad(lons)

        # Line integral based on Green's Theorem, assumes spherical Earth

        # close polygon
        if lats[0] != lats[-1]:
            lats = append(lats, lats[0])
            lons = append(lons, lons[0])

        # colatitudes relative to (0,0)
        a = sin(lats / 2) ** 2 + cos(lats) * sin(lons / 2) ** 2
        colat = 2 * arctan2(sqrt(a), sqrt(1 - a))

        # azimuths relative to (0,0)
        az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2 * pi)

        # Calculate diffs
        # daz = diff(az) % (2*pi)
        daz = diff(az)
        daz = (daz + pi) % (2 * pi) - pi

        deltas = diff(colat) / 2
        colat = colat[0:-1] + deltas

        # Perform integral
        integrands = (1 - cos(colat)) * daz

        # Integrate
        area = abs(sum(integrands)) / (4 * pi)

        # Area in ratio of sphere total area:
        area = min(area, 1 - area)

        # Area in sqft:
        areaout = area * 4 * pi * radius**2

        # Area in sqm:
        if length_unit == "m":
            areaout = areaout / (3.28084**2)

        return areaout

    def _create_asset_inventory(
        self, footprints: list, attributes: dict, length_unit: str
    ) -> AssetInventory:
        """
        This method will be used by the subclasses to form the AssetInventory given footprints & atttributes

        Args:
              footprints (list):
                      footprint of each building, each entry a list of lon, lat points
              attributes (dict):
                      dict of atttributes, each value a list of attributes for each bldg
              length_unit (str):
                      indicating length unit 'ft' for example

        Returns:
              AssetInventory:
                      A building inventory for buildings in the region.

        """

        attributes["fpAreas"] = []
        for fp in footprints:
            lons = []
            lats = []
            for pt in fp:
                lons.append(pt[0])
                lats.append(pt[1])

            attributes["fpAreas"].append(
                int(self._polygon_area(lats, lons, length_unit))
            )

        # Calculate centroids of the footprints and remove footprint data that
        # does not form a polygon:

        self.footprints = []
        self.centroids = []
        ind_remove = []

        for ind, footprint in enumerate(footprints):
            if ind == 0:
                print(footprint)
            try:
                self.centroids.append(Polygon(footprint).centroid)
                self.footprints.append(footprint)
                if ind == 0:
                    print("adding", ind, footprint)
            except:
                if ind == 0:
                    print("removing", ind, footprint)
                ind_remove.append(ind)
                pass

        # Remove attribute corresponding to the removed footprints:
        for i in sorted(ind_remove, reverse=True):
            for key in attributes.keys():
                del attributes[key][i]

        # Write the footprint data into a GeoJSON file:
        inventory = AssetInventory()

        for ind, fp in enumerate(footprints):

            asset_features = {}
            for key in attributes.keys():
                attr = attributes[key][ind]
                asset_features[key] = "NA" if attr is None else attr

            inventory.add_asset(ind, fp, asset_features)

        return inventory
