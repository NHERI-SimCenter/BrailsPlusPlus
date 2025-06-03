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
# 06-02-2025

"""
This module provides a utility class for validating input data in BRAILS.

.. autosummary::

      InputValidator
"""

from typing import Any


class InputValidator:
    """
    A utility class for validating BRAILS input data.

    The InputValidator class provides static methods to ensure that the inputs
    provided to brails are of the correct type and format. It includes methods
    to validate coordinate lists, polygons, and other relevant data structures.

    Methods:
        - is_float(element: Any) -> bool: Checks if the given input_value can
            be converted to a float.
        - validate_coordinates(coordinates: list[list[float]]
            ) -> tuple[bool, str]: Validates a two-dimensional list of
            coordinates ensuring that each coordinate pair consists of two
            floats within the valid range for longitude and latitude.
    """

    @staticmethod
    def is_float(input_value: Any) -> bool:
        """
        Check if the given input_value can be converted to a float.

        Args:
            input_value (Any): The input_value to check.

        Returns:
            bool: True if the input_value can be converted to a float, False
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
    def validate_coordinates(coordinates: list[list[float]]
                             ) -> tuple[bool, str]:
        """
        Validate input for coordinates.

        Args:
            coordinates (list[list[float]]): A two-dimensional list
                representing the geometry in [[lon1, lat1], [lon2, lat2], ...,
                [lonN, latN]] format.
        Returns:
            tuple[bool, str]: A tuple containing:
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
    def is_valid_geometry(coordinates: list[list[float]]) -> bool:
        """
        Validate whether the given coordinates represent a valid geometry.

        Args:
            coordinates (list[list[float]]):
                A list of coordinate pairs.

        Returns:
            bool:
                True if the coordinates are valid, False otherwise.
        """
        return InputValidator.validate_coordinates(coordinates)[0]

    @staticmethod
    def is_point(coordinates: list[list[float]]) -> bool:
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
        if not InputValidator.is_valid_geometry(coordinates):
            return False

        return len(coordinates) == 1

    @staticmethod
    def is_linestring(coordinates: list[list[float]]) -> bool:
        """
        Determine whether the given coordinates represent a linestring.

        In BRAILS, a linestring is defined as a sequence of at least two
        coordinate pairs, and it is not closed (the first and last points are
        different). If exactly two points are provided, it is considered a
        valid linestring

        Args:
            coordinates (list[list[float]]):
                A list of coordinate pairs forming a linestring.

        Returns:
            bool:
                True if the coordinates represent a linestring, False
                otherwise.
        """
        if not InputValidator.is_valid_geometry(coordinates):
            return False

        return len(coordinates) == 2 or (len(coordinates) > 2
                                         and coordinates[0] != coordinates[-1])

    @staticmethod
    def is_multilinestring(coordinates: list[list[list[float]]]) -> bool:
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
        if not InputValidator.is_valid_geometry(coordinates):
            return False

        for linestring in coordinates:
            if not InputValidator.is_linestring(linestring):
                return False

        return True

    @staticmethod
    def is_polygon(coordinates: list[list[float]]) -> bool:
        """
        Determine whether the given coordinates represent a polygon.

        In BRAILS, a polygon is defined as a sequence of at least three
        coordinate pairs where the first and last coordinates are the same
        (closed loop).

        Args:
            coordinates (list[list[float]]):
                A list of coordinate pairs forming a polygon.

        Returns:
            bool:
                True if the coordinates represent a polygon, False otherwise.
        """
        if not InputValidator.is_valid_geometry(coordinates):
            return False

        return len(coordinates) > 2 and coordinates[0] == coordinates[-1]

    @staticmethod
    def is_multipolygon(coordinates: list[list[list[float]]]) -> bool:
        """
        Determine whether the given coordinates represent a MultiPolygon.

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
        if not InputValidator.is_valid_geometry(coordinates):
            return False

        for polygon in coordinates:
            if not InputValidator.is_polygon(polygon):
                return False

        return True
