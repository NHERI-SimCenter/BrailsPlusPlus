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
# 10-18-2025

"""
This module provides a utility class for validating input data in BRAILS.

.. autosummary::

      InputValidator
"""
import os
from typing import Any, List, Tuple
from shapely.geometry import Polygon


class InputValidator:
    """
    A utility class for validating BRAILS geospatial input data.

    This class provides static methods to validate various geospatial data
    structures such as points, linestrings, polygons, and their
    multi-geometries, ensuring conformance with geographic coordinate standards
    (longitude/latitude).

    It also includes a lightweight utility for checking whether a file path
    points to an image file based on its extension, without opening the file.

    All methods are designed to validate nested lists of floats representing
    coordinates, with specific rules for each geometry type.

    To import the :class:`InputValidator` class, use:

    .. code-block:: python

        from brails.utils import InputValidator
    """

    @staticmethod
    def is_float(input_value: Any) -> bool:
        """
        Check if the given input_value can be converted to a float.

        Args:
            input_value (Any):
                The input value to check.

        Returns:
            bool:
                ``True`` if the ``input_value`` can be converted to a ``float``
                , ``False`` otherwise.

        Examples:
            >>> InputValidator.is_float('3.14')
            True

            >>> InputValidator.is_float('abc')
            False

            >>> InputValidator.is_float(None)
            False

            >>> InputValidator.is_float(10)
            True
        """
        if input_value is None:
            return False
        try:
            float(input_value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_coordinates(
            coordinates: list[list[float]] | list[list[list[float]]]
    ) -> tuple[bool, str]:
        """
        Validate input for coordinates.

        Args:
            coordinates (list[list[float]] | list[list[list[float]]]):
                A nested list of floats representing the geometry.
                Supports:
                - Points/LineStrings/Polygons (Depth 2): ``[[lon, lat], ...]``
                - Multi-geometries (Depth 3): ``[[[lon, lat], ...], ...]``
        Returns:
            tuple[bool, str]:
                A tuple containing:

                - A boolean indicating if all coordinates are valid.
                - A message string describing any issues found, or confirming
                  validation success.
        Examples:
            >>> InputValidator.validate_coordinates([-122.4, 37.75])
            (True, 'Coordinate pair is valid.')

            >>> InputValidator.validate_coordinates([
            ...     [-122.4, 37.75],
            ...     [-122.4, 37.76],
            ...     [-122.39, 37.76],
            ...     [-122.39, 37.75],
            ...     [-122.4, 37.75]
            ... ])
            (True, 'Coordinates input is valid')

            >>> InputValidator.validate_coordinates("invalid")
            (False, 'Coordinates input is not a list')
        """
        # Check if coordinates input is a list:
        if not isinstance(coordinates, list):
            return False, 'Coordinates input is not a list'

        if not coordinates:
            return False, 'Coordinates input is empty'

        # Base case: single coordinate pair:
        if len(coordinates) == 2 and all(isinstance(
                c, float) for c in coordinates):
            lon, lat = coordinates
            if not -180 <= lon <= 180:
                return (False, f'Longitude {lon} is not a float or is '
                        'out of range (-180 to 180).')
            if not -90 <= lat <= 90:
                return (False, f'Latitude {lat} is not a float or is out'
                        ' of range (-90 to 90).')
            return True, 'Coordinate pair is valid.'

        # Recurse if the item is a list of coordinates or sub-geometries:
        for item in coordinates:
            valid, message = InputValidator.validate_coordinates(item)
            if not valid:
                return False, message

        # If all checks pass:
        return True, "Coordinates input is valid"

    @staticmethod
    def is_point(coordinates: List[List[float]]) -> bool:
        """
        Determine whether the given coordinates represent a point.

        In BRAILS, a point is defined as a single coordinate pair.

        Args:
            coordinates (list[list[float]]):
                A list containing coordinate pairs [latitude, longitude].

        Returns:
            bool:
                ``True`` if the coordinates represent a point, ``False``
                otherwise.

        Examples:
            >>> InputValidator.is_point([[-122.4, 37.75]])
            True

            >>> InputValidator.is_point([
            ...     [-122.4, 37.75],
            ...     [-122.39, 37.76]
            ... ])
            False
        """
        if not InputValidator.validate_coordinates(coordinates)[0]:
            return False

        return len(coordinates) == 1

    @staticmethod
    def is_linestring(coordinates: List[List[float]]) -> bool:
        """
        Determine whether the input represents a valid BRAILS linestring.

        In BRAILS, a valid linestring must:

        - Be a list of coordinate pairs, where each pair is a ``list`` of
          exactly two ``float`` values: [longitude, latitude].
        - Contain at least two coordinate pairs.
        - Not be a nested structure beyond one level (i.e., must be a ``list``
          of ``list``, not a ``list`` of ``list`` of ``list``).
        - The first and last coordinate pairs must not be the same (i.e., the
          linestring must not be closed, to avoid confusion with a polygon).

        This function strictly rejects inputs with excessive nesting, such as
        multilinestrings (i.e., ``list`` of ``list`` of coordinate pairs).

        Args:
            coordinates (list[list[float]]):
                A list of coordinate pairs forming a linestring.

        Returns:
            bool:
                ``True`` if the coordinates represent a linestring, ``False``
                otherwise.

        Examples:
            >>> InputValidator.is_linestring([
            ...     [-122.4, 37.75],
            ...     [-122.39, 37.76]
            ... ])
            True

            >>> InputValidator.is_linestring([
            ...     [-122.4, 37.75],
            ...     [-122.39, 37.76],
            ...     [-122.4, 37.75]
            ... ])
            False

            >>> InputValidator.is_linestring([
            ...     [[-122.4, 37.75], [-122.39, 37.76]]
            ... ])
            False
        """
        if not InputValidator.validate_coordinates(coordinates)[0]:
            return False

        return (len(coordinates) >= 2
                and all(
                    isinstance(pair, list)
                    and len(pair) == 2
                    and all(isinstance(coord, float) for coord in pair)
                    for pair in coordinates
        )
            and coordinates[0] != coordinates[-1]
        )

    @staticmethod
    def is_multilinestring(coordinates: List[List[List[float]]]) -> bool:
        """
        Determine whether the given coordinates represent a multilinestring.

        In BRAILS, a multilinestring is defined as a ``list`` of valid
        linestrings, each with at least 2 points and not forming a closed loop.

        Args:
            coordinates (list[list[list[float]]]):
                A list of linestrings, each represented as a list of
                coordinate pairs.

        Returns:
            bool:
                ``True`` if the coordinates represent a multilinestring,
                ``False`` otherwise.

        Examples:
            >>> InputValidator.is_multilinestring([
            ...     [
            ...         [-122.4, 37.75],
            ...         [-122.4, 37.76],
            ...         [-122.39, 37.76],
            ...         [-122.39, 37.75]
            ...     ],
            ...     [
            ...         [-122.38, 37.74],
            ...         [-122.38, 37.75],
            ...         [-122.37, 37.75],
            ...         [-122.37, 37.74]
            ...     ]
            ... ])
            True

            >>> InputValidator.is_multilinestring([
            ...     [-122.4, 37.75],
            ...     [-122.4, 37.76],
            ...     [-122.39, 37.76],
            ...     [-122.39, 37.75]
            ... ])
            False
        """
        if not InputValidator.validate_coordinates(coordinates)[0]:
            return False

        for linestring in coordinates:
            if not InputValidator.is_linestring(linestring):
                return False

        return True

    @staticmethod
    def is_polygon(coordinates: List[List[float]]) -> bool:
        """
        Determine whether the input represents a valid BRAILS polygon geometry.

        A valid polygon must:

        - Be a ``list`` of coordinate pairs.
        - Each coordinate pair must be a ``list`` of exactly two ``float``
          values: [longitude, latitude].
        - Contain at least three coordinate pairs, plus a fourth that closes
          the shape.
        - The first and last coordinate pairs must be the same
          (i.e., the polygon must form a closed loop).
        - The structure must be exactly one level deep: a ``list`` of
          coordinate pairs, not a ``list`` of ``list`` of coordinate
          ``list``.

        This function performs structural and type checks and uses
        :meth:`validate_coordinates()` to ensure coordinate validity.

        Args:
            coordinates (list[list[float]]):
                A list of coordinate pairs forming a polygon.

        Returns:
            bool:
                ``True`` if the coordinates represent a polygon, ``False``
                otherwise.

        Examples:
            >>> InputValidator.is_polygon([
            ...     [-122.4, 37.75],
            ...     [-122.4, 37.76],
            ...     [-122.39, 37.76],
            ...     [-122.39, 37.75],
            ...     [-122.4, 37.75]
            ... ])
            True

            >>> InputValidator.is_polygon([
            ...     [-122.4, 37.75],
            ...     [-122.4, 37.76],
            ...     [-122.39, 37.76],
            ...     [-122.39, 37.75]
            ... ])
            False
        """
        if not InputValidator.validate_coordinates(coordinates)[0]:
            return False

        return (len(coordinates) >= 2
                and all(
                    isinstance(pair, list)
                    and len(pair) == 2
                    and all(isinstance(coord, float) for coord in pair)
                    for pair in coordinates
        )
            and coordinates[0] == coordinates[-1]
        )

    @staticmethod
    def is_multipolygon(coordinates: List[List[List[float]]]) -> bool:
        """
        Determine whether given coordinates represent a BRAILS multipolygon.

        In BRAILS, a multipolygon is defined as a ``list`` of valid polygons,
        where each polygon is a closed loop with at least 3 points.

        Args:
            coordinates (list[list[list[float]]]):
                A list of polygons, each represented as a list of coordinate
                pairs.

        Returns:
            bool:
                ``True`` if the coordinates represent a multipolygon, ``False``
                otherwise.

        Examples:
            >>> InputValidator.is_multipolygon([
            ...     [
            ...         [-122.4, 37.75],
            ...         [-122.4, 37.76],
            ...         [-122.39, 37.76],
            ...         [-122.39, 37.75],
            ...         [-122.4, 37.75]
            ...     ],
            ...     [
            ...         [-122.38, 37.74],
            ...         [-122.38, 37.75],
            ...         [-122.37, 37.75],
            ...         [-122.37, 37.74],
            ...         [-122.38, 37.74]
            ...     ]
            ... ])
            True

            >>> InputValidator.is_multipolygon([
            ...     [-122.4, 37.75],
            ...     [-122.4, 37.76],
            ...     [-122.39, 37.76],
            ...     [-122.39, 37.75]
            ... ])
            False
        """
        if not InputValidator.validate_coordinates(coordinates)[0]:
            return False

        for polygon in coordinates:
            if not InputValidator.is_polygon(polygon):
                return False

        return True

    @staticmethod
    def is_image(filepath: str) -> bool:
        """
        Perform a lightweight check to determine if the file is an image.

        This function checks that the path exists, is a file, and has a valid
        image file extension. It does not open or validate the contents of the
        file, so it cannot detect corrupted or mislabeled files.

        Args:
            filepath (str):
                The path to the file to check.

        Returns:
            bool:
                ``True`` if the ``filepath`` points to a file with a supported
                image extension, otherwise ``False``.

        Examples:
            >>> InputValidator.is_image('example.jpg')  # Assuming file exists
            True

            >>> InputValidator.is_image('document.pdf')
            False

            >>> InputValidator.is_image('/path/to/missing.png')
            False
        """
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        return os.path.isfile(filepath) and filepath.lower().endswith(
            valid_exts)

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
                A Shapely Polygon to be checked.

        Returns:
            bool:
                ``True`` if the ``geometry`` is a rectangular box, ``False``
                otherwise.

        Raises:
            TypeError:
                If the input is not a Shapely Polygon


        Examples:
            >>> from shapely.geometry import Polygon
            >>> # A rectangle:
            >>> rect = Polygon([(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)])
            >>> GeoTools.is_box(rect)
            True

            >>> # A non-rectangle polygon:
            >>> poly = Polygon([(0, 0), (1, 0), (2, 1), (0, 1), (0, 0)])
            >>> GeoTools.is_box(poly)
            False
        """
        # Check if the input is a polygon:
        if not isinstance(geometry, Polygon):
            raise TypeError(
                'Invalid geometry input. Expected a Shapely Polygon object.'
            )

        # Check if the geometry has exactly 4 corners:
        coords = list(geometry.exterior.coords)
        if len(coords) != 5:
            return False

        # Extract points:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4), _ = coords

        # Check if opposite sides are equal (box property):
        return (x1 == x4 and x2 == x3 and y1 == y2 and y3 == y4)
