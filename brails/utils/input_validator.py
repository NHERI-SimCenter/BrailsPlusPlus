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
# 08-05-2025

"""
This module provides a utility class for validating input data in BRAILS.

.. autosummary::

      InputValidator
"""
import os
from typing import Any, List, Tuple


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

    Methods:
        is_float(input_value: Any) -> bool:
            Checks whether a given value can be safely converted to a float.
        validate_coordinates(coordinates: List[List[float]])->Tuple[bool, str]:
            Validates a nested list of coordinates, ensuring correct structure
            and ranges for longitude (-180 to 180) and latitude (-90 to 90).
        is_valid_geometry(coordinates: List[List[float]]) -> bool:
            Returns whether the input coordinates pass the base validation
            check.
        is_point(coordinates: List[List[float]]) -> bool:
            Returns True if the coordinates represent a valid single-point
            geometry.
        is_linestring(coordinates: List[List[float]]) -> bool:
            Returns True if the coordinates represent a valid LineString (at
            least two distinct points and not closed).
        is_multilinestring(coordinates: List[List[List[float]]]) -> bool:
            Returns True if the coordinates represent a MultiLineString
            composed of valid LineStrings.
        is_polygon(coordinates: List[List[float]]) -> bool:
            Returns True if the coordinates form a valid Polygon (closed loop
            with at least 3 points).
        is_multipolygon(coordinates: List[List[List[float]]]) -> bool:
            Returns True if the coordinates form a valid MultiPolygon (a list
            of valid Polygons).
        is_image(filepath: str) -> bool:
            Returns True if the given path points to a file that exists and has
            a valid image file extension ('.jpg', '.jpeg', '.png', '.bmp').
            This method does not read or load the image file.
    """

    @staticmethod
    def is_float(input_value: Any) -> bool:
        """
        Check if the given input_value can be converted to a float.

        Args:
            input_value (Any):
                The input_value to check.

        Returns:
            bool:
                True if the input_value can be converted to a float, False
                otherwise.
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
            coordinates: List[List[float]]
    ) -> Tuple[bool, str]:
        """
        Validate input for coordinates.

        Args:
            coordinates (list[list[float]]):
                A two-dimensional list
                representing the geometry in [[lon1, lat1], [lon2, lat2], ...,
                [lonN, latN]] format.
        Returns:
            tuple[bool, str]:
                A tuple containing:
                - A boolean indicating if all coordinates are valid.
                - A message string describing any issues found, or confirming
                    validation success.
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
                True if the coordinates represent a point, False otherwise.
        """
        if not InputValidator.validate_coordinates(coordinates)[0]:
            return False

        return len(coordinates) == 1

    @staticmethod
    def is_linestring(coordinates: List[List[float]]) -> bool:
        """
        Determine whether the input represents a valid BRAILS LineString.

        In BRAILS, a valid LineString must:
        - Be a list of coordinate pairs, where each pair is a list of exactly
          two float values: [longitude, latitude].
        - Contain at least two coordinate pairs.
        - Not be a nested structure beyond one level (i.e., must be a list of
          lists, not a list of list of lists).
        - The first and last coordinate pairs must not be the same (i.e., the
          LineString must not be closed, to avoid confusion with a Polygon).

        This function strictly rejects inputs with excessive nesting, such as
        MultiLineStrings (i.e., lists of list of coordinate pairs).

        Args:
            coordinates (list[list[float]]):
                A list of coordinate pairs forming a linestring.

        Returns:
            bool:
                True if the coordinates represent a linestring, False
                otherwise.
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
        Determine whether the given coordinates represent a MultiLineString.

        In BRAILS, a MultiLineString is defined as a list of valid linestrings,
        each with at least 2 points and not forming a closed loop.

        Args:
            coordinates (list[list[list[float]]]):
                A list of linestrings, each represented as a list of coordinate
                pairs.

        Returns:
            bool:
                True if the coordinates represent a MultiLineString, False
                otherwise.
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
        Determine whether the input represents a valid BRAILS Polygon geometry.

        A valid Polygon must:
        - Be a list of coordinate pairs.
        - Each coordinate pair must be a list of exactly two float values:
            [longitude, latitude].
        - Contain at least three coordinate pairs, plus a fourth that closes
          the shape.
        - The first and last coordinate pairs must be the same
          (i.e., the polygon must form a closed loop).
        - The structure must be exactly one level deep: a list of coordinate
          pairs, not a list of lists of coordinate lists.

        This function performs structural and type checks and uses
        `validate_coordinates()` to ensure coordinate validity.

        Args:
            coordinates (list[list[float]]):
                A list of coordinate pairs forming a polygon.

        Returns:
            bool:
                True if the coordinates represent a polygon, False otherwise.
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
        Determine whether given coordinates represent a BRAILS MultiPolygon.

        In BRAILS, a MultiPolygon is defined as a list of valid polygons,
        where each polygon is a closed loop with at least 3 points.

        Args:
            coordinates (list[list[list[float]]]):
                A list of polygons, each represented as a list of coordinate
                pairs.

        Returns:
            bool:
                True if the coordinates represent a MultiPolygon, False
                otherwise.
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
                True if the path points to a file with a supported image
                extension, otherwise False.
        """
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        return os.path.isfile(filepath) and filepath.lower().endswith(
            valid_exts)
