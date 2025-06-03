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

    GetPointsInPolygons
"""
from __future__ import annotations
from shapely.strtree import STRtree
from shapely.geometry import Point, Polygon
from brails.utils.spatial_join_methods.base import SpatialJoinMethods
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory


class GetPointsInPolygons(SpatialJoinMethods):
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
        print('\nJoining inventories...')
        matched_polygon_ids, matched_point_ids = \
            self._find_points_in_polygons(polygon_inventory,
                                          point_inventory)

        print(f'Identified a total of {len(matched_polygon_ids)} matched '
              'points.')
        for polygon_id, point_id in zip(
                matched_polygon_ids, matched_point_ids):
            point_features = point_inventory.inventory[point_id].features
            polygon_inventory.add_asset_features(
                polygon_id, point_features
            )
        print('Inventories successfully joined.')

        return polygon_inventory

    def _find_points_in_polygons(self,
                                 polygon_inventory: AssetInventory,
                                 point_inventory: AssetInventory):
        """
        Perform a spatial join to find which points lie within which polygons.

        For each polygon in the polygon inventory, this method identifies the
        point(s) from the point inventory that fall within it. If multiple
        points are found within a polygon, only the point closest to the
        polygon's centroid is selected. The result is a mapping of matched
        point IDs to polygon IDs.

        Args:
            polygon_inventory (AssetInventory):
                Inventory with polygon geometric data
                asset IDs.
            point_inventory (AssetInventory):
                Inventory with point geometric data

        Returns:
            tuple:
                A tuple containing:
                - matched_polygon_ids (list[str | int]):
                    Asset IDs of polygons that have at least one matched point.
                - matched_point_ids (list[str | int]):
                    Asset IDs of points that matched with a polygon.

        Process:
            1. Extract asset IDs and coordinates from both inventories.
            2. Construct shapely Polygon objects from the polygon coordinates.
            3. Construct shapely Point objects from the point coordinates.
            4. Build an STRtree spatial index on the points for efficient
               lookup.
            5. For each polygon:
                a. Query the STRtree for points inside the polygon.
                b. If multiple points match, select the one closest to the
                   polygon's centroid.
                c. Map the matched point ID to the polygon ID.
            6. Return the matched polygon and point IDs.
        """
        polygon_asset_ids = self._get_polygon_indices(polygon_inventory)
        point_asset_ids = self._get_point_indices(point_inventory)

        coordinates, asset_ids = polygon_inventory.get_coordinates()
        polygons = [coordinates[asset_ids.index(
            asset_id)] for asset_id in polygon_asset_ids]

        coordinates, asset_ids = point_inventory.get_coordinates()
        points = [Point(coordinates[asset_ids.index(
            asset_id)][0]) for asset_id in point_asset_ids]

        # Create an STR tree for the input points:
        pttree = STRtree(points)

        # Initialize the dictionary mapping point keys to polygons keys:
        fps_matched = {}

        for ind, poly in enumerate(polygons):
            polygon = Polygon(poly)

            # Query points that are within the polygon:
            res = pttree.query(polygon)

            if res.size > 0:  # Check if any points were found:
                # If multiple points exist in polygon, find the closest to the
                # centroid:
                if res.size > 1:
                    source_points = pttree.geometries.take(res)
                    poly_centroid = polygon.centroid
                    nearest_point = min(source_points,
                                        key=poly_centroid.distance)
                    nearest_index = next((index for index, point in
                                          enumerate(source_points) if
                                          point.equals(nearest_point)), None)
                    res = [nearest_index]

                fps_matched[polygon_asset_ids[ind]] = point_asset_ids[res[0]]
        return list(fps_matched.keys()), list(fps_matched.values())
