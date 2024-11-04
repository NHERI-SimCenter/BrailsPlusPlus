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
# 11-03-2024

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
            return False, 'Coordinates input is not a 2D list.'

        # Validate each coordinate pair:
        for coord in coordinates:
            # Check if each coordinate entry is a list consisting of two
            # elements:
            if not isinstance(coord, list) or len(coord) != 2:
                return (False, 'Coordinates input is not a 2D list. Each '
                        'coordinate entry should be a list with exactly two '
                        'float elements.')

            # Check longitude validity:
            if not isinstance(coord[0], float) or not -180 <= coord[0] <= 180:
                return (False, f'Longitude {coord[0]} is not a float or is '
                        'out of range (-180 to 180).')

            # Check latitude validity
            if not isinstance(coord[1], float) or not -90 <= coord[1] <= 90:
                return (False, f'Latitude {coord[1]} is not a float or is out'
                        ' of range (-90 to 90).')

        # If all checks pass:
        return True, "Coordinates input is valid"
