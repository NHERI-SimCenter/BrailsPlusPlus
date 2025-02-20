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
# 02-20-2025

"""
This module defines a class of methods for spatial joins.

.. autosummary::

      SpatialJoinMethods
"""

from shapely.strtree import STRtree
from shapely.geometry import Polygon


class SpatialJoinMethods:
    """
    A utility class containing spatial join operations for AssetInventories.

    This class provides static methods for spatial joins, where points are
    matched to polygons, and the closest points are selected when multiple
    points fall within the same polygon. The results include both the matched
    points and the polygon-point correspondence.

    Methods:
        get_points_in_polygons(points: list, polygons: list) -> AssetInventory:
            Matches points to polygons and returns the correspondence data.

        execute(method_name: str) -> any:
            Executes the spatial join method specified by `method_name` and
            returns its result.
    """

    @staticmethod
    def get_points_in_polygons(points: list,
                               polygons: list) -> tuple[list, dict, list]:
        """
        Match points to polygons and return the correspondence data.

        This function finds Shapely points that fall within given polygons.
        If multiple points exist within a polygon, it selects the point closest
        to the polygon's centroid.

        Args:
            points (list): A list of Shapely points.
            polygons (list): A list of polygon coordinates defined in
                             EPSG 4326 (longitude, latitude).

        Returns:
            tuple[list, dict, list]:
                - A list of matched Shapely Point geometries.
                - A dictionary mapping each polygon (represented as a string)
                    to the corresponding matched point.
                - A list of the indices of polygons matched to points, with
                    each index listed in the same order as the list of points.
        """
        # Create an STR tree for the input points:
        pttree = STRtree(points)

        # Initialize the list to keep indices of matched points and the mapping
        # dictionary:
        ptkeepind = []
        fp2ptmap = {}
        ind_fp_matched = []

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

                # Add the found point(s) to the keep index list:
                ptkeepind.extend(res)

                # Map polygon to the first matched point:
                fp2ptmap[str(poly)] = points[res[0]]

                # Store the index of the matched polygon:
                ind_fp_matched.append(ind)

        # Convert the list of matched points to a set for uniqueness and back
        # to a list:
        ptkeepind = list(set(ptkeepind))

        # Create a list of points that includes just the points that have a
        # polygon match:
        ptskeep = [points[ind] for ind in ptkeepind]

        return ptskeep, fp2ptmap, ind_fp_matched

    @staticmethod
    def execute(method_name):
        """
        Execute a spatial join method based on the method name.

        This method allows you to dynamically execute a spatial join method
        from the `SpatialJoinMethods` class by passing the method's name as a
        string.

        Args:
            method_name (str):
                The name of the spatial join method to execute.

        Returns:
            any:
                The result of the executed method.

        Raises:
            AttributeError: If the method specified by `method_name` is not
            found.

        Example:
            result = SpatialJoinMethods.execute('get_points_in_polygons')
        """
        # Get method from class
        method = getattr(SpatialJoinMethods, method_name, None)
        if callable(method):  # Check if it's a valid method
            return method()
        else:
            available_methods = [func for func in dir(SpatialJoinMethods)
                                 if callable(getattr(SpatialJoinMethods, func))
                                 and not func.startswith("__")]
            raise AttributeError(f"Method '{method_name}' not found. Available"
                                 f" methods: {available_methods}")
