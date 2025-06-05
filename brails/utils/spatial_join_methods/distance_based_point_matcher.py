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
# 06-05-2025

"""
This module defines the concrete class GetPointsInPolygons.

.. autosummary::

    DistanceBasedPointMatcher
"""
from __future__ import annotations
from shapely.geometry import Point, Polygon
from brails.utils.spatial_join_methods.base import SpatialJoinMethods
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory


class DistanceBasedPointMatcher(SpatialJoinMethods):
    """
    A spatial join strategy that matches points based on Euclidean distance.

    This is useful in spatial matching tasks where no containment relationship
    exists, and proximity is the primary criterion for association.

    Inherits:
        SpatialJoinMethods:
            Base class providing shared spatial join functionality.

    Methods:
        _join_implementation(receiving_point_inventory,
                             merging_point_inventory):
            Performs the proximity-based join and merges attributes.
    """

    def _join_implementation(self,
                             receiving_point_inventory: AssetInventory,
                             merging_point_inventory: AssetInventory
                             ) -> AssetInventory:
        """
        Match each point with the closest point and merge their attributes.

        Args:
            receiving_point_inventory (AssetInventory):
                Inventory containing the points that will receive attributes.
            merging_point_inventory (AssetInventory):
                Inventory containing the points to be matched and merged.

        Returns:
            AssetInventory:
                Updated point inventory with merged attributes.
        """
        receiving_pt_ids = self._get_polygon_indices(receiving_point_inventory)
        merging_pt_ids = self._get_point_indices(merging_point_inventory)

        print(f'\nJoining inventories using {self.__class__.__name__} '
              'method...')

        receiving_pt_coords, polygon_ids = \
            receiving_point_inventory.get_coordinates()
        receiving_pt_lookup = dict(zip(polygon_ids, receiving_pt_coords))
        polygons = {asset_id: Polygon(receiving_pt_lookup[asset_id])
                    for asset_id in receiving_pt_ids}

        marging_point_coords, point_ids = \
            merging_point_inventory.get_coordinates()
        merging_point_lookup = dict(zip(point_ids, marging_point_coords))
        points = {asset_id: Point(merging_point_lookup[asset_id][0])
                  for asset_id in merging_pt_ids}

        matched_points = self._find_closest_points_to_polygons(
            polygons,
            points)

        print(f'Identified a total of {len(matched_points)} matched '
              'points.')

        # Merge attributes from merging points into receiving points:
        receiving_point_inventory = self._merge_inventory_features(
            receiving_point_inventory,
            merging_point_inventory,
            matched_points
        )
        print('Inventories successfully joined.')
        return receiving_point_inventory

    def _find_closest_point_to_point(
            self,
            points: dict[int | str, Point],
            matching_points: dict[int | str, Point]
    ) -> dict[int | str, int | str]:
        """
        Find the closest matching point to each input point.

        Args:
            points (dict[int | str, Point]):
                Dictionary mapping point IDs from receiving inventory
                to Shapely Point objects.
            matching_points (dict[int | str, Point]):
                Dictionary mapping point IDs from merging inventory
                to Shapely Point objects.

        Returns:
            dict[int | str, int | str]:
                A dictionary mapping each receiving point ID to the closest
                merging point ID.
        """
        point_to_closest_point = {}

        for pt_id, pt_geometry in points.items():
            closest_point_id = min(points.items(),
                                   key=lambda item: pt_geometry.distance(
                                       item[1]))[0]
            point_to_closest_point[pt_id] = closest_point_id

        return point_to_closest_point
