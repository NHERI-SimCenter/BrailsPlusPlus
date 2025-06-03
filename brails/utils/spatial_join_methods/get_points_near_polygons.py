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
# 02-25-2025

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
    Class that implements a spatial join method for finding points in polygons.

    A class that implements a spatial join method for finding correspondence
    between points and polygons. Specifically, this class identifies points
    that fall within polygons.

    Inherits from the SpatialJoinMethods class, which likely contains
    common functionality for spatial joins.

    Methods:
        join_implementation(polygon_inventory, point_inventory):
            Joins points and polygons based on spatial relationships.

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

        print('\nJoining inventories...')
        join_instance = GetPointsInPolygons()
        matched_polygon_ids, matched_point_ids = \
            join_instance._find_points_in_polygons(polygon_inventory,
                                                   point_inventory)

        for polygon_id, point_id in zip(
                matched_polygon_ids, matched_point_ids):
            point_features = point_inventory.inventory[point_id].features
            polygon_inventory.add_asset_features(
                polygon_id, point_features
            )

        unmatched_polygon_ids = list(set(polygon_asset_ids) -
                                     set(matched_polygon_ids))

        coordinates, asset_ids = polygon_inventory.get_coordinates()
        polygons = [Polygon(coordinates[asset_ids.index(
            asset_id)]) for asset_id in unmatched_polygon_ids]

        coordinates, asset_ids = point_inventory.get_coordinates()
        points = [Point(coordinates[asset_ids.index(
            asset_id)][0]) for asset_id in point_asset_ids]

        matched_polygons = self._find_closest_points_to_polygons(
            polygons,
            unmatched_polygon_ids,
            points,
            point_asset_ids)

        n_matched_points = len(matched_polygon_ids) + len(matched_polygons)
        print(f'Identified a total of {n_matched_points} matched '
              'points.')

        for polygon_id, point_id in matched_polygons.items():
            point_features = point_inventory.inventory[point_id].features
            polygon_inventory.add_asset_features(
                polygon_id, point_features
            )
        print('Inventories successfully joined.')
        return polygon_inventory

    def _find_closest_points_to_polygons(self,
                                         polygons: list[Polygon],
                                         polygon_ids: list[int | str],
                                         points: list[Point],
                                         point_ids: list[int | str]):
        """
        Find the closest point (from a list) to the centroid of each polygon.

        Args:
            polygons (list[Polygon]):
                List of Shapely Polygon objects.
            points (list[Point]):
                List of Shapely Point objects.

        Returns:
            dict:
                A dictionary mapping each polygon index to the closest Point
                object.
        """
        polygon_to_closest_point = {}

        for i, polygon in enumerate(polygons):
            centroid = polygon.centroid
            closest_point = min(points, key=lambda pt: centroid.distance(pt))
            closest_point_index = points.index(closest_point)
            closest_point_id = point_ids[closest_point_index]
            polygon_to_closest_point[polygon_ids[i]] = closest_point_id

        return polygon_to_closest_point
