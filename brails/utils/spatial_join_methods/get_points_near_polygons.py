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
# 06-04-2025

"""
This module defines the concrete class GetPointsInPolygons.

.. autosummary::

    GetPointsNearPolygons
"""
from __future__ import annotations
from shapely.geometry import Point, Polygon
from brails.utils.spatial_join_methods.base import SpatialJoinMethods
from brails.utils.spatial_join_methods.get_points_in_polygons import \
    GetPointsInPolygons
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory


class GetPointsNearPolygons(SpatialJoinMethods):
    """
    A spatial join strategy that associates point features with polygons.

    This class performs a two-stage spatial join:

    1. **Containment-Based Join:** Points that lie inside a polygon are matched
       directly.
    2. **Proximity-Based Join:** For polygons without any internal points, the
       closest point to the polygon's centroid is selected.

    The result is a polygon inventory enriched with features from matched point
    geometries.

    Inherits:
        SpatialJoinMethods:
            Base class providing shared spatial join functionality.

    Methods:
        _join_implementation(polygon_inventory, point_inventory):
            Executes the spatial join and feature merge logic.
    """

    def _join_implementation(self,
                             polygon_inventory: AssetInventory,
                             point_inventory: AssetInventory):
        """
        Join associating point features with polygons they fall within.

        For each polygon that contains a point, the point's features are added
        to the corresponding polygon asset in the inventory.

        Args:
            polygon_inventory (AssetInventory):
                Inventory of polygon geometries.
            point_inventory (AssetInventory):
                Inventory of point geometries.

        Returns:
            AssetInventory:
                Updated polygon inventory with merged point features.
        """
        polygon_asset_ids = self._get_polygon_indices(polygon_inventory)
        point_asset_ids = self._get_point_indices(point_inventory)

        print(f'\nJoining inventories using {self.__class__.__name__} '
              'method...')
        join_instance = GetPointsInPolygons()
        matched_polygons1 = join_instance._find_points_in_polygons(
            polygon_inventory,
            point_inventory
        )

        # Merge attributes from points into polygons:
        polygon_inventory = self._merge_inventory_features(polygon_inventory,
                                                           point_inventory,
                                                           matched_polygons1)

        unmatched_polygon_ids = list(set(polygon_asset_ids) -
                                     set(matched_polygons1.keys()))

        polygon_coords, polygon_ids = polygon_inventory.get_coordinates()
        polygon_lookup = dict(zip(polygon_ids, polygon_coords))
        polygons = {asset_id: Polygon(polygon_lookup[asset_id])
                    for asset_id in unmatched_polygon_ids}

        point_coords, point_ids = point_inventory.get_coordinates()
        point_lookup = dict(zip(point_ids, point_coords))
        points = {asset_id: Point(point_lookup[asset_id][0])
                  for asset_id in point_asset_ids}

        matched_polygons2 = self._find_closest_points_to_polygons(
            polygons,
            points)

        n_matched_points = len(matched_polygons1) + len(matched_polygons2)
        print(f'Identified a total of {n_matched_points} matched '
              'points.')

        # Merge attributes from points into polygons:
        polygon_inventory = self._merge_inventory_features(polygon_inventory,
                                                           point_inventory,
                                                           matched_polygons2)
        print('Inventories successfully joined.')
        return polygon_inventory

    def _find_closest_points_to_polygons(
            self,
            polygons: dict[int | str, Polygon],
            points: dict[int | str, Point]) -> dict[int | str, int | str]:
        """
        Find the closest point to the centroid of each polygon.

        Args:
            polygons (dict[int | str, Polygon]):
                Dictionary mapping polygon IDs to Shapely Polygon objects.
            points (dict[int | str, Point]):
                Dictionary mapping point IDs to Shapely Point objects.

        Returns:
            dict[int | str, int | str]:
                A dictionary mapping each polygon ID to closest point ID.
        """
        polygon_to_closest_point = {}

        for poly_id, poly in polygons.items():
            centroid = poly.centroid
            closest_point_id = min(points.items(),
                                   key=lambda item: centroid.distance(
                                       item[1]))[0]
            polygon_to_closest_point[poly_id] = closest_point_id

        return polygon_to_closest_point
