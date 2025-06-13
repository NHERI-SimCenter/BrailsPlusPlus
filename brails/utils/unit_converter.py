# Copyright (c) 2025 The Regents of the University of California
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
# 06-13-2025

"""
This module defines a class for performing unit conversions.

.. autosummary::

    UnitConverter
"""
UNIT_TO_METER = {
    'm': 1.0,
    'km': 1000.0,
    'cm': 0.01,
    'mm': 0.001,
    'in': 0.0254,
    'ft': 0.3048,
    'yd': 0.9144,
    'mi': 1609.344
}


class UnitConverter:
    """
    A utility class for converting between different units of length and area.

    This class provides static methods to:
    - Convert between supported length units (e.g., meters to feet).
    - Convert between supported area units (e.g., square kilometers to square
      yards).

    Supported length units:
        'm', 'km', 'cm', 'mm', 'in', 'ft', 'yd', 'mi'

    Supported area units:
        'm2', 'km2', 'cm2', 'mm2', 'in2', 'ft2', 'yd2', 'mi2'

    Methods:
        print_supported_units() -> None
            Print the list of supported length and area units to the console.
        convert_length(value: float, from_unit: str, to_unit: str) -> float:
            Convert a length value from one unit to another.
        convert_area(value: float, from_unit: str, to_unit: str) -> float:
            Convert an area value from one unit to another.

    Note:
        All conversions are based on SI metric definitions.
        Area conversions are computed using squared base length conversions.

    Usage Example:
        UnitConverter.convert_length(10, 'm', 'ft')
        UnitConverter.convert_area(100, 'm2', 'ft2')
        UnitConverter.print_supported_units()
    """

    @staticmethod
    def print_supported_units() -> None:
        """Print supported units."""
        print("Supported length units:", ', '.join(UNIT_TO_METER.keys()))
        print("Supported area units:  ",
              ', '.join(f"{u}2" for u in UNIT_TO_METER.keys()))

    @staticmethod
    def convert_length(value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert a length value from one unit to another.

        Supported units: 'm', 'km', 'cm', 'mm', 'in', 'ft', 'yd', 'mi'

        Args:
            value (float):
                The numeric value to convert.
            from_unit (str):
                The current unit of the value.
            to_unit (str):
                The desired unit to convert to.

        Returns:
            float:
                Converted value in the target unit.

        Raises:
            ValueError:
                If either unit is unsupported.
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        if from_unit not in UNIT_TO_METER or to_unit not in UNIT_TO_METER:
            raise ValueError('Unsupported units. Supported units are: '
                             f'{list(UNIT_TO_METER.keys())}')

        value_in_meters = value * UNIT_TO_METER[from_unit]
        converted_value = value_in_meters / UNIT_TO_METER[to_unit]

        return converted_value

    @staticmethod
    def convert_area(value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert an area value from one unit to another.

        Supported units: 'm2', 'km2', 'cm2', 'mm2', 'in2', 'ft2', 'yd2', 'mi2'

        Args:
            value (float):
                The numeric area value to convert.
            from_unit (str):
                The current area unit (e.g., 'm2').
            to_unit (str):
                The desired area unit (e.g., 'ft2').

        Returns:
            float:
                Converted area value in the target unit.

        Raises:
            ValueError:
                If either unit is unsupported.
        """
        # Strip the '2' from area units to get base length units:
        from_unit_base = from_unit.lower().rstrip('2')
        to_unit_base = to_unit.lower().rstrip('2')

        if (from_unit_base not in UNIT_TO_METER or
                to_unit_base not in UNIT_TO_METER):
            raise ValueError('Unsupported units. Supported units are: '
                             f"{[f'{u}2' for u in UNIT_TO_METER.keys()]}")

        from_factor = UNIT_TO_METER[from_unit_base] ** 2
        to_factor = UNIT_TO_METER[to_unit_base] ** 2

        value_in_m2 = value * from_factor
        converted_value = value_in_m2 / to_factor

        return converted_value
