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
# 06-06-2025

"""
This module defines a class for geospatial analysis and operations.

.. autosummary::

      GeoTools
"""

import json
from math import radians, sin, cos, atan2, sqrt
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from shapely import to_geojson
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.strtree import STRtree

# Constants:
R_EARTH_KM = 6371.0
KM2_FEET = 3280.84


class GeoTools:
    """
    A collection of static methods for geospatial analysis and operations.

    The GeoTools class provides various utility functions to perform common
    geospatial tasks, including distance calculations, polygon meshing,
    plotting, and GeoJSON file handling. The methods leverage Shapely
    geometries to operate on points and polygons, making it suitable for
    geographical data manipulation and visualization.

    Methods:
        haversine_dist(p1: tuple, p2: tuple) -> float:
          Calculate the Haversine distance between two geographical points.
        mesh_polygon(polygon: Polygon, rows: int, cols: int) ->
            list[Polygon]:
            Split a polygon into a grid of individual rectangular polygons.
        plot_polygon_cells(bpoly: Polygon | MultiPolygon,
            rectangles: list[Polygon], output_file: str = ''):
            Plot a polygon/its rectangular mesh, optionally saving the plot.
        write_polygon_to_geojson(poly: Polygon | MultiPolygon,
            output_file: str):
            Write a Shapely Polygon or MultiPolygon to a GeoJSON file.
        match_points_to_polygons(points: list, polygons: list) ->
            tuple[list, dict]:
            Match points to polygons and return the correspondence data.
    """

    @staticmethod
    def haversine_dist(
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> float:
        """
        Calculate the Haversine distance between two points.

        This function computes the shortest distance over the earth's surface
        between two points specified by their latitude and longitude.

        Args:
            p1 (tuple):
                The first point as a list containing two floating-point
                values, where the first element is the latitude and the second
                is the longitude of the point in degrees.
            p2 (tuple):
                The second point as a list containing two floating-point
                values, where the first element is the latitude and the second
                is the longitude of the point in degrees.

        Returns:
            float:
                The Haversine distance between the two points in feet.
        """
        # Convert coordinate values from degrees to radians:
        lat1, lon1 = radians(p1[0]), radians(p1[1])
        lat2, lon2 = radians(p2[0]), radians(p2[1])

        # Compute the difference between latitude and longitude values:
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        # Compute the distance between two points as a proportion of Earth's
        # mean radius:
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Return distance between the two points in feet:
        return R_EARTH_KM * c * KM2_FEET

    @staticmethod
    def mesh_polygon(polygon: Polygon, rows: int, cols: int) -> List[Polygon]:
        """
        Split a Spolygon into a grid of individual rectangular polygons.

        This function divides the area of the input polygon into a specified
        number of rows and columns, creating a mesh of rectangular cells.
        Each cell is checked for intersection with the input polygon, and
        only the intersecting parts are returned.

        Args:
            polygon (Polygon):
                A Shapely polygon to be meshed.
            rows (int):
                The number of rows to divide the polygon into.
            cols (int):
                The number of columns to divide the polygon into.

        Returns:
            list[Polygon]:
                A list of Shapely polygons representing the individual
                rectangular cells that mesh the input polygon.
        """
        # Get bounds of the polygon:
        min_x, min_y, max_x, max_y = polygon.bounds

        # Calculate dimensions of each cell:
        width = (max_x - min_x) / cols
        height = (max_y - min_y) / rows

        rectangles = []
        # For each cell:
        for i in range(rows):
            for j in range(cols):
                # Calculate coordinates of the cell vertices:
                x1 = min_x + j * width
                y1 = min_y + i * height
                x2 = x1 + width
                y2 = y1 + height

                # Convert the computed geometry into a polygon:
                cell_polygon = Polygon(
                    [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

                # Check if the obtained cell intersects with the polygon:
                if cell_polygon.intersects(polygon):
                    intersection = cell_polygon.intersection(polygon)
                    # If the intersection is a finite geometry keep its
                    # envelope as a valid cell:
                    if intersection.is_empty:
                        continue
                    rectangles.append(intersection.envelope)

        return rectangles

    @staticmethod
    def plot_polygon_cells(bpoly: Union[Polygon, MultiPolygon],
                           rectangles: List[Polygon],
                           output_file: str = ''):
        """
        Plot a polygon and its rectangular mesh, optionally saving the plot.

        Args:
            bpoly (Polygon | MultiPolygon):
                A Shapely polygon or MultiPolygon to plot.
            rectangles (list[Polygon]):
                A list of Shapely polygons representing the rectangular cells
                that mesh input polygon.
            output_file (str, optional):
                Filename to save the plot as a PNG image. If empty string, the
                plot is not saved.

        Raises:
            ValueError:
                If `output_file` is provided and an invalid filename is given.
        """
        # Plot the base polygon:
        if bpoly.geom_type == "MultiPolygon":
            for poly in bpoly.geoms:
                plt.plot(*poly.exterior.xy, color='black')
        else:
            plt.plot(*bpoly.exterior.xy, color='black')

        # Plot the rectangular cells:
        for rect in rectangles:
            # Check if the rectangle is valid before plotting
            if not rect.is_empty:
                plt.plot(*rect.exterior.xy, color='blue')

        # Save the plot if a filename is provided:
        if output_file:
            try:
                plt.savefig(output_file, dpi=600, bbox_inches="tight")
            except Exception as e:
                raise ValueError(f"Error saving the file: {e}") from e

        plt.show()

    @staticmethod
    def write_polygon_to_geojson(poly: Union[Polygon, MultiPolygon],
                                 output_file: str):
        """
        Write a Shapely Polygon or MultiPolygon to a GeoJSON file.

        Args:
            poly (Polygon | MultiPolygon):
                A Shapely polygon or MultiPolygon to be written.
            output_file (str):
                The output filename for the GeoJSON file.

        """
        if 'geojson' not in output_file.lower():
            output_file = output_file.replace(
                output_file.split('.')[-1], 'geojson')

        # Create the GeoJSON structure:
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": []
        }

        # Determine the geometry type and coordinates:
        polytype = poly.geom_type
        coordinates = json.loads(
            to_geojson(poly).split('"coordinates":')[-1][:-1])

        feature = {"type": "Feature",
                   "properties": {},
                   "geometry": {"type": polytype,
                                "coordinates": coordinates
                                }
                   }

        geojson["features"].append(feature)

        # Write the GeoJSON to the specified file:
        with open(output_file, 'w', encoding='utf8') as outfile:
            json.dump(geojson, outfile, indent=2)

    @staticmethod
    def match_points_to_polygons(
        points: List[Point],
        polygons: List[Polygon]
    ) -> Tuple[List[Point], Dict[str, Point], List[int]]:
        """
        Match points to polygons and return the correspondence data.

        This function finds Shapely points that fall within given polygons.
        If multiple points exist within a polygon, it selects the point closest
        to the polygon's centroid.

        Args:
            points (list):
                A list of Shapely points.
            polygons (list):
                A list of polygon coordinates defined in EPSG 4326
                (longitude, latitude).

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
    def is_box(geometry: Polygon) -> bool:
        """
        Determine whether a given Shapely geometry is a rectangular box.

        A box is defined as a Polygon with exactly four corners and opposite
        sides being equal. This function checks if the geometry is a Polygon
        with 5 coordinates (the 5th being a duplicate of the first to close the
        polygon), and verifies that opposite sides are equal, ensuring that the
        polygon is rectangular.

        Args:
            geometry (Polygon):
                A Shapely Polygon object to be checked.

        Returns:
            bool:
                True if the Polygon is a rectangular box, False otherwise.

        Raises:
            TypeError:
                If the input is not a Shapely Polygon object.
        """
        # Check if the input is a polygon:
        if not isinstance(geometry, Polygon):
            TypeError('Invalid geometry input. Expected a Shapely Polygon '
                      'object.')

        # Check if the geometry has exactly 4 corners:
        if len(geometry.exterior.coords) == 5:
            # Check if opposite sides are equal (box property):
            x1, y1 = geometry.exterior.coords[0]
            x2, y2 = geometry.exterior.coords[1]
            x3, y3 = geometry.exterior.coords[2]
            x4, y4 = geometry.exterior.coords[3]

            return (x1 == x2 and y1 == y4 and x3 == x4 and y2 == y3)
        return False
