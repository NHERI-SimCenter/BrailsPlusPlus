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
# 07-15-2025

"""
This module defines a class for performing unit parsing and conversions.

.. autosummary::

    UnitConverter
"""
# Implemented unit types:
LENGTH = 'length'
AREA = 'area'
WEIGHT = 'weight'

# Conversion factors from meters:
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

# Conversion factors from kilograms:
UNIT_TO_KG = {
    'mg': 0.000001,
    'g': 0.001,
    'kg': 1.0,
    'ton': 1000.0,        # metric ton
    'oz': 0.0283495,
    'lb': 0.453592,
    'ton_us': 907.18474,  # US ton
    'ton_uk': 1016.04691  # UK ton
}

# Common aliases for the implemented units:
UNIT_ALIASES = {
    # Length
    'meter': 'm', 'meters': 'm',
    'kilometer': 'km', 'kilometers': 'km',
    'centimeter': 'cm', 'centimeters': 'cm',
    'millimeter': 'mm', 'millimeters': 'mm',
    'inch': 'in', 'inches': 'in',
    'foot': 'ft', 'feet': 'ft',
    'yard': 'yd', 'yards': 'yd',
    'mile': 'mi', 'miles': 'mi',

    # Area
    'sqm': 'm2', 'sqft': 'ft2', 'sqft.': 'ft2',
    'sqin': 'in2', 'sqyd': 'yd2', 'sqkm': 'km2', 'sqmi': 'mi2',
    'square meter': 'm2', 'square foot': 'ft2', 'square feet': 'ft2',
    'square kilometer': 'km2', 'square mile': 'mi2',

    # Weight
    'milligram': 'mg', 'milligrams': 'mg',
    'gram': 'g', 'grams': 'g',
    'kilogram': 'kg', 'kilograms': 'kg',
    'metric ton': 'ton', 'metric tons': 'ton',
    'ounce': 'oz', 'ounces': 'oz',
    'pound': 'lb', 'pounds': 'lb', 'lbs': 'lb',
    'us ton': 'ton_us', 'short ton': 'ton_us',
    'uk ton': 'ton_uk', 'long ton': 'ton_uk'
}


class UnitConverter:
    """
    A utility class for converting values between different units.

    This class provides static methods for unit conversion across three
    categories:
    - Length (e.g., meters to feet)
    - Area (e.g., square kilometers to square yards)
    - Weight (e.g., kilograms to pounds)

    Supported unit types and examples:
        Length units:
            'm', 'km', 'cm', 'mm', 'in', 'ft', 'yd', 'mi'

        Area units (derived from squared base length units):
            'm2', 'km2', 'cm2', 'mm2', 'in2', 'ft2', 'yd2', 'mi2'

        Weight units:
            - Metric: 'mg', 'g', 'kg', 'ton' (metric ton = 1,000 kg)
            - Imperial:
                'oz'       = ounce
                'lb'       = pound
                'ton_us'   = US short ton (2,000 lb)
                'ton_uk'   = UK long ton (2,240 lb)

    Key Features:
        - Supports standard and alias unit names (e.g., "meter", "m", "kg").
        - Automatically validates unit category compatibility.
        - Converts via standardized base units:
            - Meters for length
            - Square meters for area
            - Kilograms for weight

    Methods:
        convert_length(value, from_unit, to_unit) -> float:
            Convert a numeric value between supported length units.
        convert_area(value, from_unit, to_unit) -> float:
            Convert a numeric value between supported area units (e.g., m2 to
            ft2).
        convert_weight(value, from_unit, to_unit) -> float:
            Convert a numeric value between supported weight units (e.g., kg to
            lb).
        convert_unit(value, from_unit, to_unit) -> float:
            Auto-detect the category (length, area, or weight) and convert
            between compatible units.
        normalize_unit(unit) -> str:
            Standardize unit strings and resolve common aliases (e.g., "meters"
            is "m").
        get_unit_type(unit) -> str:
            Return the category of a unit ('length', 'area', or 'weight').
        unit_valid(unit, expected_unit_type) -> bool:
            Check if a unit belongs to the expected category and raise an error
            if not.
        parse_units(input_dict, unit_defaults) -> dict:
            Extract and validate units from a dictionary, applying defaults
            where needed.
        get_supported_units(unit_type='all') -> str:
            Return a formatted string listing supported units for a given type
            or all types.

    Notes:
        - Area conversions are based on the square of length conversions.
        - Weight conversion uses kilograms as the internal reference unit.
        - An error is raised if units are mismatched or unrecognized.

    Example Usage:
        >>> UnitConverter.convert_length(10, 'm', 'ft')
        32.8084

        >>> UnitConverter.convert_area(100, 'm2', 'ft2')
        1076.391041671

        >>> UnitConverter.convert_weight(1, 'ton', 'lb')
        2204.6226218487757

        >>> UnitConverter.convert_unit(5, 'km', 'mi')
        3.106855

        >>> UnitConverter.convert_unit(2500, 'g', 'lb')
        5.51156

        >>> UnitConverter.normalize_unit('Meters')
        'm'

        >>> UnitConverter.get_unit_type('yd2')
        'area'

        >>> UnitConverter.unit_valid('oz', 'weight')
        True

        >>> UnitConverter.parse_units({'weight': 'lb'},
                                      {'length': 'm', 'weight': 'kg'})
        {'length': 'm', 'weight': 'lb'}

        >>> print(UnitConverter.get_supported_units())
        Length: cm, ft, in, km, m, mi, mm, yd
        Area:   cm2, ft2, in2, km2, m2, mi2, mm2, yd2
        Weight: g, kg, lb, mg, oz, ton, ton_uk, ton_us

    """

    def get_supported_units(unit_type: str = 'all') -> str:
        """
        Return a formatted string listing supported units for a unit type.

        Args:
            unit_type (str):
                The type of unit to retrieve ('length', 'area', 'weight', or
                'all'). Defaults to 'all'.

        Returns:
            str:
                A human-readable string of supported units.
        """
        unit_type = unit_type.lower()

        length_units = ', '.join(sorted(UNIT_TO_METER.keys()))
        area_units = ', '.join(sorted(f"{u}2" for u in UNIT_TO_METER.keys()))
        weight_units = ', '.join(sorted(UNIT_TO_KG.keys()))

        unit_map = {
            LENGTH: f"  Length: {length_units}",
            AREA: f"  Area:   {area_units}",
            WEIGHT: f"  Weight: {weight_units}"
        }

        if unit_type == 'all':
            return 'Supported Units: \n' + "\n".join(unit_map.values()) + "\n"
        elif unit_type in unit_map:
            return unit_map[unit_type] + "\n"
        else:
            supported_types = ', '.join(sorted(unit_map.keys()) + ['all'])
            return (f"Error: Invalid unit type '{unit_type}'. Supported types "
                    f"are: {supported_types}.\n"
                    )

    @staticmethod
    def normalize_unit(unit: str):
        """
        Normalize unit name by reformatting and resolving aliases.

        This function ensures consistent internal representation of units,
        allowing users to use common names or variations (e.g., "meters",
        "LBS", "square foot") which will be mapped to standard short forms
        (e.g., "m", "lb", "ft2").

        Args:
            unit (str):
                The input unit string to normalize.

        Returns:
            str:
                The standardized unit string. If no alias is found, returns the
                cleaned input.
        """
        unit = unit.strip().lower()
        return UNIT_ALIASES.get(unit, unit)

    @staticmethod
    def get_unit_type(unit: str):
        """
        Identify unit type as 'length', 'area', or 'weight'.

        Returns:
            str:
                Unit type.
        Raises:
            ValueError:
                If unit is not recognized.
        """
        if unit.endswith('2') and unit[:-1] in UNIT_TO_METER:
            return AREA
        elif unit in UNIT_TO_METER:
            return LENGTH
        elif unit in UNIT_TO_KG:
            return WEIGHT
        else:
            unit_list = UnitConverter.get_supported_units('all')
            raise ValueError(
                f"Unsupported or unknown unit: '{unit}'. \n"
                f"{unit_list}"
            )

    @staticmethod
    def unit_valid(unit: str, expected_unit_type: str) -> bool:
        """
        Validate that a unit belongs to the expected unit type.

        Args:
            unit (str):
                The unit to check (e.g., 'm', 'kg', 'ft2').
            expected_unit_type (str):
                The expected type ('length', 'area', or 'weight').

        Returns:
            bool:
                True if the unit matches the expected type. False otherwise
        """
        actual_unit_type = UnitConverter.get_unit_type(unit)
        if actual_unit_type == expected_unit_type:
            return True
        return False

    @staticmethod
    def parse_units(input_dict: dict, unit_defaults: dict) -> dict:
        """
        Parse and validate units from input_dict based on defaults.

        Args:
            input_dict (dict):
                Dictionary with unit inputs.
            unit_defaults (dict):
                Default unit values for each unit type.

        Returns:
            dict:
                Validated units per type (e.g., {'length': 'ft',
                'weight': 'lb'}).

        Raises:
            ValueError:
                If a unit is invalid.
        """
        parsed_units = {}
        for unit_type, default in unit_defaults.items():
            raw_value = input_dict.get(unit_type)
            unit_value = (raw_value or default).lower()

            if raw_value is None:
                print(f"No {unit_type} unit specified. Using default: "
                      f"'{default}'.")

            if not UnitConverter.unit_valid(unit_value, unit_type):
                supported_units = UnitConverter.get_supported_units(unit_type)
                raise ValueError(
                    f"Incorrect {unit_type} unit defined: '{unit_value}'. "
                    f"Supported {unit_type} units: {supported_units}"
                )

            parsed_units[unit_type] = unit_value
        return parsed_units

    @staticmethod
    def convert_length(value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert a length value from one unit to another.

        Supported units: 'm', 'km', 'cm', 'mm', 'in', 'ft', 'yd', 'mi'

        Args:
            value(float):
                The numeric value to convert.
            from_unit(str):
                The current unit of the value.
            to_unit(str):
                The desired unit to convert to.

        Returns:
            float:
                Converted value in the target unit.

        Raises:
            ValueError:
                If either unit is unsupported.
        """
        from_unit = UnitConverter.normalize_unit(from_unit)
        to_unit = UnitConverter.normalize_unit(to_unit)

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
            value(float):
                The numeric area value to convert.
            from_unit(str):
                The current area unit(e.g., 'm2').
            to_unit(str):
                The desired area unit(e.g., 'ft2').

        Returns:
            float:
                Converted area value in the target unit.

        Raises:
            ValueError:
                If either unit is unsupported.
        """
        # Strip the '2' from area units to get base length units:
        from_unit_base = UnitConverter.normalize_unit(from_unit).rstrip('2')
        to_unit_base = UnitConverter.normalize_unit(to_unit).rstrip('2')

        if (from_unit_base not in UNIT_TO_METER or
                to_unit_base not in UNIT_TO_METER):
            raise ValueError('Unsupported units. Supported units are: '
                             f"{[f'{u}2' for u in UNIT_TO_METER.keys()]}")

        from_factor = UNIT_TO_METER[from_unit_base] ** 2
        to_factor = UNIT_TO_METER[to_unit_base] ** 2

        value_in_m2 = value * from_factor
        converted_value = value_in_m2 / to_factor

        return converted_value

    @staticmethod
    def convert_weight(value: float, from_unit: str, to_unit: str):
        """
        Convert a weight value from one unit to another.

        Supported units: 'mg', 'g', 'kg', 'ton' (metric ton), 'oz', 'lb',
        'ton_us', 'ton_uk'

        Args:
            value(float):
                The numeric value to convert.
            from_unit(str):
                The source unit.
            to_unit(str):
                The target unit.

        Returns:
            float:
                Converted value in the target unit.

        Raises:
            ValueError:
                If either unit is unsupported.
        """
        from_unit = UnitConverter.normalize_unit(from_unit)
        to_unit = UnitConverter.normalize_unit(to_unit)

        if from_unit not in UNIT_TO_KG or to_unit not in UNIT_TO_KG:
            raise ValueError(
                "Unsupported units. Supported units are: {}".format(list(
                    UNIT_TO_KG.keys()))
            )

        value_in_kg = value * UNIT_TO_KG[from_unit]
        converted_value = value_in_kg / UNIT_TO_KG[to_unit]

        return converted_value

    @staticmethod
    def convert_unit(value: float, from_unit: str, to_unit: str):
        """
        Convert a numerical value between compatible units.

        This method supports conversions across three unit categories: length,
        area, and weight. Unit aliases (e.g., "kg" for "kilogram") are
        supported. The function automatically detects the category of the input
        and output units and ensures they are compatible.

        Args:
            value (float):
                The numerical value to convert.
            from_unit (str):
                The source unit (e.g., "meter", "sqft", "kg"). Aliases are
                supported.
            to_unit (str):
                The target unit for conversion. Must be in the same category as
                `from_unit`.

        Returns:
            float:
                The value converted to the target unit.

        Raises:
            ValueError:
                If the units belong to different categories (e.g., length vs
                weight), are unrecognized, or if the unit type is unsupported.
        """
        from_unit = UnitConverter.normalize_unit(from_unit)
        to_unit = UnitConverter.normalize_unit(to_unit)

        from_type = UnitConverter.get_unit_type(from_unit)
        to_type = UnitConverter.get_unit_type(to_unit)

        if from_type != to_type:
            raise ValueError(
                f"Incompatible unit types: cannot convert from '{from_type}' "
                f"to '{to_type}'. Units must belong to the same category "
                "(e.g., both 'length')."

            )

        if from_type == LENGTH:
            return UnitConverter.convert_length(value, from_unit, to_unit)
        elif from_type == AREA:
            return UnitConverter.convert_area(value, from_unit, to_unit)
        elif from_type == WEIGHT:
            return UnitConverter.convert_weight(value, from_unit, to_unit)

        raise ValueError("Unhandled unit type: {}".format(from_type))
