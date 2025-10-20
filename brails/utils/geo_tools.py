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
# 10-20-2025

"""
This module defines a class for geospatial analysis and operations.

.. autosummary::

      GeoTools
"""

import json
from math import radians, sin, cos, atan2, sqrt
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
from shapely import to_geojson
from shapely.geometry import box, LineString, MultiLineString, MultiPolygon, \
    Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from brails.constants import R_EARTH_KM
from brails.utils.unit_converter import UnitConverter
from brails.utils.input_validator import InputValidator

class GeoTools:
    """
    A collection of static methods for geospatial analysis and operations.

    The GeoTools class provides various utility functions to perform common
    geospatial tasks, including distance calculations, polygon meshing,
    plotting, and GeoJSON file handling. The methods leverage Shapely
    geometries to operate on points and polygons, making it suitable for
    geographical data manipulation and visualization.

    To import the :class:`GeoTools` class, use:

    .. code-block:: python

        from brails.utils import GeoTools
    """

    @staticmethod
    def bbox2poly(
        query_area: Tuple[float, ...],
        output_file: str = ''
    ) -> Tuple[Polygon, str]:
        """
        Get the boundary polygon for a region based on its coordinates.

        This method parses the provided bounding polygon coordinates into a
        polygon object. The polygon  must be defined by at least two pairs of
        longitude/latitude values (i.e., at least 4 elements) and must have an
        even number of elements.  If a file name is provided in the `outfile`
        argument, the resulting polygon is saved to a GeoJSON file.

        Args:
            query_area (tuple):
                A tuple containing longitude/latitude pairs that define a
                bounding box. The tuple should contain at least two pairs of
                coordinates (i.e., 4 values), and the number of elements must
                be an even number.
            output_file (str, optional):
                If a file name is provided, the resulting polygon will be
                written to the specified file in GeoJSON format.

        Raises:
            TypeError:
                If ``query_area`` is not a tuple.
            ValueError:
                If the tuple has an odd number of elements or fewer than two
                pairs.

        Returns:
            Tuple[Polygon, str]:

                - The bounding polygon as a Shapely Polygon.
                - A human-readable string representation of the bounding
                  polygon.

        Example:
            Simple bounding box (two coordinate pairs):

            >>> from brails.utils import GeoTools
            >>> coords = (-122.3, 37.85, -122.25, 37.9)
            >>> bpoly, description = GeoTools.bbox2poly(coords)
            >>> print(bpoly)
            POLYGON ((-122.3 37.85, -122.3 37.9, -122.25 37.9, -122.25 37.85,
                      -122.3 37.85))
            >>> print(description)
            the bounding box: (-122.3, 37.85, -122.25, 37.9)

            A triangular polygon:

            >>> long_coords = (
            ...     -122.3, 37.85, -122.28, 37.86, -122.26, 37.87, -122.25,
            ...     37.9
            ... )
            >>> bpoly_long, desc_long = GeoTools.bbox2poly(long_coords)
            >>> print(bpoly_long)
            POLYGON ((-122.3 37.85, -122.28 37.86, -122.26 37.87, -122.25 37.9,
            -122.3 37.85))
            >>> print(desc_long)
            the bounding polygon: [(-122.3, 37.85), (-122.28, 37.86),
            (-122.26, 37.87), (-122.25, 37.9)]
        """
        if not isinstance(query_area, tuple):
            raise TypeError(
                'Query area must be a tuple of longitude/latitude values.'
            )

        n_coords = len(query_area)

        if n_coords < 4 or n_coords % 2 != 0:
            raise ValueError(
                'Bounding polygon must be defined as tuple consisting of at'
                'least two longitude/latitude pairs and an even number of '
                'elements.'
            )

        # Convert tuple into list of coordinate pairs
        coords = [(query_area[i], query_area[i + 1])
                  for i in range(0, n_coords, 2)]

        # Create Shapely polygon
        bpoly = box(*query_area) if n_coords == 4 else Polygon(coords)

        # Build human-readable description
        if n_coords == 4:
            queryarea_printname = f'the bounding box: {query_area}'
        else:
            queryarea_printname = f'the bounding polygon: {coords}'

        # Write to file if requested
        if output_file:
            GeoTools.write_polygon_to_geojson(bpoly, output_file)

        return bpoly, queryarea_printname

    @staticmethod
    def geometry_to_list_of_lists(
            geom: BaseGeometry
            ) -> Union[List[List[float]], List[List[List[float]]]]:
        """
        Convert a Shapely geometry into a list of coordinate lists.

        Args:
            geom (BaseGeometry):
                A Shapely geometry (such as Point, Polygon)

        Returns:
            List[List[float]] or List[List[List[float]]]:
                A list of ``[lon, lat]`` values or nested list objects
                for complex geometries.

        Examples:
            >>> from shapely.geometry import Point, Polygon
            >>> GeoTools.geometry_to_list_of_lists(Point(1.0, 2.0))
            [[1.0, 2.0]]
            >>> square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
            >>> GeoTools.geometry_to_list_of_lists(square)
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        """
        if isinstance(geom, Point):
            return [[geom.x, geom.y]]

        elif isinstance(geom, LineString):
            return [list(coord) for coord in geom.coords]

        elif isinstance(geom, MultiLineString):
            return [
                [list(coord) for coord in line.coords] for line in geom.geoms
            ]

        elif isinstance(geom, Polygon):
            return [list(coord) for coord in geom.exterior.coords]

        elif isinstance(geom, MultiPolygon):
            return [
                [list(coord) for coord in polygon.exterior.coords]
                for polygon in geom.geoms
            ]

        else:
            raise TypeError(f"Unsupported geometry type: {type(geom)}")

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
            float: The Haversine distance between the two points in feet.

        Example:
            >>> p1 = (37.7749, -122.4194)  # San Francisco, CA
            >>> p2 = (34.0522, -118.2437)  # Los Angeles, CA
            >>> distance = GeoTools.haversine_dist(p1, p2)
            >>> round(distance / 5280)  # convert feet to miles
            347
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
        return UnitConverter.convert_length(R_EARTH_KM * c, 'km', 'ft')
    
    @staticmethod
    def list_of_lists_to_geometry(
            coordinates: Union[List[List[float]], List[List[List[float]]]]
            ) -> BaseGeometry:
        """
        Convert BRAILS nested coordinate lists into a Shapely geometry.

        Uses the existing ``InputValidator`` helpers to decide which geometry
        to build. Assumes coordinates are in WGS-84 order 
        ``[longitude, latitude]``.

        Args:
            coordinates (list[list[float]] | list[list[list[float]]]):
                - Point: ``[[lon, lat]]``
                - LineString: ``[[lon, lat], [lon, lat], ...]``
                - Polygon (single exterior ring): ``[[lon, lat], ..., [lon, lat]]`` (closed)
                - MultiLineString: ``[[[lon, lat], ...], [[lon, lat], ...], ...]``
                - MultiPolygon (exterior rings only): ``[[[lon, lat], ...], ...]``
                  where each inner list is a **closed** polygon ring.

        Returns:
            shapely.geometry.BaseGeometry: 
                One of ``Point``, ``LineString``, ``Polygon``,
                ``MultiLineString``, or ``MultiPolygon``.

        Raises:
            ValueError: 
                If the coordinates do not match any permitted geometry type per
                ``InputValidator``.

        Examples:
            >>> GeoTools.list_of_lists_to_geometry([[-122.3321, 47.6062]])
            <POINT (-122.332 47.606)>
            
            >>> GeoTools.list_of_lists_to_geometry([
            ...     [-122.335, 47.606],
            ...     [-122.334, 47.607],
            ...     [-122.333, 47.606]
            ... ])
            <LINESTRING (-122.335 47.606, -122.334 47.607, -122.333 47.606)>
            
            >>> polygon = [
            ...     [-122.336, 47.606],
            ...     [-122.334, 47.606],
            ...     [-122.334, 47.608],
            ...     [-122.336, 47.608],
            ...     [-122.336, 47.606]
            ... ]
            >>> GeoTools.list_of_lists_to_geometry(polygon)
            <POLYGON ((-122.336 47.606, -122.334 47.606, -122.334 47.608,
            -122.336 47.608, -122.336 47.606))>
            
            >>> multiline = [
            ...     [[-122.337, 47.606], [-122.335, 47.607]],
            ...     [[-122.334, 47.607], [-122.333, 47.608], [-122.332, 47.607]]
            ... ]
            >>> GeoTools.list_of_lists_to_geometry(multiline)
            <MULTILINESTRING ((-122.337 47.606, -122.335 47.607), 
            (-122.334 47.607, -122.333 47.608, -122.332 47.607))>
            
            >>> multipolygon = [
            ...     [
            ...         [-122.337, 47.606],
            ...         [-122.335, 47.606],
            ...         [-122.335, 47.608],
            ...         [-122.337, 47.608],
            ...         [-122.337, 47.606]
            ...     ],
            ...     [
            ...         [-122.334, 47.607],
            ...         [-122.332, 47.607],
            ...         [-122.332, 47.609],
            ...         [-122.334, 47.609],
            ...         [-122.334, 47.607]
            ...     ]
            ... ]
            >>> GeoTools.list_of_lists_to_geometry(multipolygon)
            <MULTIPOLYGON (((-122.337 47.606, -122.335 47.606, -122.335 47.608,
            -122.337 47.608, -122.337 47.606)), ((-122.334 47.607, 
            -122.332 47.607, -122.332 47.609, -122.334 47.609, 
            -122.334 47.607)))>
        """        
    
        if InputValidator.is_point(coordinates):
            return Point(*coordinates[0])
        
        if InputValidator.is_linestring(coordinates):
            return LineString(coordinates)
        
        if InputValidator.is_polygon(coordinates):
            return Polygon(coordinates)

        if InputValidator.is_multilinestring(coordinates):
            return MultiLineString(coordinates)

        if InputValidator.is_multipolygon(coordinates):        
            polygons = [Polygon(ring) for ring in coordinates]
            return MultiPolygon(polygons)

        raise ValueError(
            'Unsupported coordinate structure: expected a nested list '
            'conforming to BRAILS geometry specifications for one of the '
            'following types: Point, LineString, MultiLineString, Polygon, '
            'or MultiPolygon.'
        )

        


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
            points (list[Point]):
                A list of Shapely points
            polygons (list[Polygon]):
                A list of Shapely Polygons defined in EPSG 4326
                (longitude, latitude).

        Returns:
            tuple[list, dict, list]:

                - A list of matched Shapely points.
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
    def mesh_polygon(polygon: Polygon, rows: int, cols: int) -> List[Polygon]:
        """
        Split a Spolygon into a grid of individual rectangular polygons.

        This function divides the area of the input polygon into a specified
        number of rows and columns, creating a mesh of rectangular cells.
        Each cell is checked for intersection with the input polygon, and
        only the intersecting parts are returned.

        Args:
            polygon (Polygon):
                A Shapely Polygon to be meshed.
            rows (int):
                The number of rows to divide the polygon into.
            cols (int):
                The number of columns to divide the polygon into.

        Returns:
            list[Polygon]:
                A list of Shapely Polygons representing the individual
                rectangular cells that mesh the ``polygon`` input.

        Example:
            >>> from shapely.geometry import Polygon
            >>> poly = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
            >>> cells = GeoTools.mesh_polygon(poly, rows=2, cols=2)
            >>> len(cells)
            4
            >>> # Access coordinates of first cell
            >>> cells[0].bounds
            (0.0, 0.0, 2.0, 1.5)
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
    def parse_geojson_geometry(
            geometry: Dict[str, Any]
            ) -> Union[List[List[float]], List[List[List[float]]]]:
        """
        Convert a GeoJSON geometry object into BRAILS asset coordinates.

        This function standardizes coordinates of different GeoJSON geometry
        types, such as ``Point``, ``MultiPoint``, ``LineString``, 
        ``MultiLineString``, ``Polygon``, and ``MultiPolygon`` into BRAILS
        representation: a nested list of ``[x, y]`` coordinate pairs.
        
        Depending on the geometry type, the return value may represent either
        a single sequence of coordinate pairs (``list[list[float]]``) or a
        collection of multiple coordinate sequences
        (``list[list[list[float]]]``).

        Args:
            geometry (dict):  
                A GeoJSON geometry dictionary containing at least the keys
                ``"type"`` and ``"coordinates"``, e.g., ``{"type": "Polygon",
                "coordinates": [[[x1, y1], [x2, y2], ...]]}``

        Returns:
            list[list[float]] or list[list[list[float]]]:  
                - ``list[list[float]]`` for geometries with a single coordinate
                  list (e.g., Point, LineString, Polygon).
                - ``list[list[list[float]]]`` for geometries composed of 
                  multiple coordinate sequences (e.g., MultiLineString, 
                  MultiPolygon).

        Raises:
            TypeError: If ``geometry`` is not a dictionary.  
            ValueError: If the geometry is missing required keys.  
            NotImplementedError: If the geometry type is unsupported.
        
        Examples:
            >>> GeoTools.parse_geojson_geometry(
            ...     {"type": "Point", "coordinates": [10.0, 20.0]}
            ... )
            [[10.0, 20.0]]
    
            >>> GeoTools.parse_geojson_geometry(
            ...     {"type": "LineString", 
                     "coordinates": [[0.0, 0.0], [1.0, 1.0]]
                     }
            ... )
            [[0.0, 0.0], [1.0, 1.0]]
    
            >>> GeoTools.parse_geojson_geometry(
            ...     {"type": "MultiLineString",
            ...      "coordinates": [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]}
            ... )
            [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]
    
            >>> GeoTools.parse_geojson_geometry(
            ...     {"type": "Polygon",
            ...      "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
            ... )
            [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    
            >>> GeoTools.parse_geojson_geometry(
            ...     {"type": "MultiPolygon",
            ...      "coordinates": [
            ...          [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            ...          [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
            ...      ]}
            ... )
            [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
             [[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
        """
        if not isinstance(geometry, dict):
            raise TypeError('Geometry must be a dictionary')
        if 'type' not in geometry or 'coordinates' not in geometry:
            raise ValueError(
                "Invalid geometry: missing 'type' or 'coordinates' keys"
                )
       
        gtype = geometry['type']
        coords = geometry['coordinates']
       
        if gtype == 'Point':
            return [coords]
       
        elif gtype in ['LineString', 'MultiLineString']:
            return coords
              
        elif gtype == 'Polygon':
            # Exterior ring only
            return coords[0]
       
        elif gtype == 'MultiPolygon':
            # Flatten all exterior rings
            return [poly[0] for poly in coords]
       
        else:
            raise NotImplementedError(
                f'Unsupported GeoJSON geometry type: {gtype}'
                )

    @staticmethod
    def plot_polygon_cells(
            bpoly: Union[Polygon, MultiPolygon],
            rectangles: List[Polygon],
            output_file: str = ''
    ) -> None:
        """
        Plot a polygon and its rectangular mesh, optionally saving the plot.

        This function visualizes a Shapely Polygon or MultiPolygon along with
        its rectangular mesh of cells. Each cell is plotted in blue, and the
        base polygon is outlined in black.

        Args:
            bpoly (Polygon or MultiPolygon):
                A Shapely Polygon or MultiPolygon to plot.
            rectangles (list[Polygon]):
                A list of Shapely Polygon objects representing the rectangular
                cells that mesh input polygon.
            output_file (str, optional):
                Filename to save the plot as a PNG image. If empty string, the
                plot is not saved.

        Returns:
            None

        Raises:
            ValueError:
                If ``output_file`` is provided and an invalid filename is
                given.

        Example:
            >>> from shapely.geometry import Polygon
            >>> poly = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
            >>> cells = GeoTools.mesh_polygon(poly, rows=2, cols=2)
            >>> GeoTools.plot_polygon_cells(poly, cells)
            >>> GeoTools.plot_polygon_cells(
            ...     poly,
            ...     cells,
            ...     output_file='mesh_plot.png'
            ... )
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
    def write_polygon_to_geojson(
        poly: Union[Polygon, MultiPolygon],
        output_file: str
    ):
        """
        Write a Shapely Polygon or MultiPolygon to a GeoJSON file.

        Args:
            poly (Polygon or MultiPolygon):
                A Shapely Polygon or MultiPolygon to be written.
            output_file (str):
                The output filename for the GeoJSON file.

        Note:
            - This method does not perform validation on the geometry.
            - The file extension will be replaced with '.geojson' if not
              present.

        Returns:
            None

        Examples:
            >>> from shapely.geometry import Polygon, MultiPolygon
            >>> poly = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
            >>> GeoTools.write_polygon_to_geojson(poly, 'my_polygon.geojson')

            >>> poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
            >>> poly2 = Polygon([(3, 0), (5, 0), (5, 2), (3, 2)])
            >>> mpoly = MultiPolygon([poly1, poly2])
            >>> GeoTools.write_polygon_to_geojson(
            ...     mpoly,
            ...     'my_multipolygon.geojson'
            ... )
        """
        if 'geojson' not in output_file.lower():
            output_file = output_file.rsplit('.', 1)[0] + '.geojson'

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