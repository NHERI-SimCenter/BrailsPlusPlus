# Copyright (c) 2026 The Regents of the University of California
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
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.

"""
Unit counts of housing records and the occupancy classes they give.

.. autosummary::

      count_band
      unit_contribution
      point_class
      stated_number
"""

from __future__ import annotations

import math

import numpy as np

# The housing classes, the classes whose records carry a unit count: the
# single-family, mobile home and multi-family classes of the Hazus Inventory
# Technical Manual, Hazus 7.0 (FEMA, June 2025), Table 4-1, p. 4-2, which Hazus 6.1
# prints alike in its own Table 4-1. The table's other residential classes, RES4 to
# RES6, carry no unit count here. The two year samplers, the unit budget, the pair
# re-draw and the observed-donor imputer's default status read the same tuple as
# their residential classes.
RES3_CLASSES = ('RES3A', 'RES3B', 'RES3C', 'RES3D', 'RES3E', 'RES3F')
UNIT_COUNT_CLASSES = ('RES1', 'RES2', *RES3_CLASSES)

# Lower bound of each count band, the unit ranges of the occupancy classes in the
# Hazus Inventory Technical Manual, Hazus 7.0 (FEMA, June 2025), Table 4-1, p. 4-2. A
# unit count falls in the band whose lower bound it has reached.
COUNT_BANDS = (
    (50, 'RES3F'),
    (20, 'RES3E'),
    (10, 'RES3D'),
    (5, 'RES3C'),
    (3, 'RES3B'),
    (2, 'RES3A'),
    (0, 'RES1'),
)

# Families whose class is the count band of the unit count. A RES2 record keeps RES2
# whatever its unit count.
COUNTED_FAMILIES = ('RES1', 'RES3')

# Units a housing record contributes when its resunits is zero, negative or missing:
# the midpoint of its class's unit band, one dwelling for RES1 and RES2, and a fixed
# count for the open RES3F band.
UNIT_MIDPOINTS = {
    'RES1': 1.0,
    'RES2': 1.0,
    'RES3A': 2.0,
    'RES3B': 3.5,
    'RES3C': 7.0,
    'RES3D': 14.5,
    'RES3E': 34.5,
    'RES3F': 100.0,
}


def stated_number(value: object) -> float | int | None:
    """
    Return a feature value that states a finite number, or None.

    Args:
        value (object):
            A raw feature value.

    Returns:
        float or int or None:
            The value itself where it is a finite int or float (a bool is not),
            otherwise None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    number = value.item() if isinstance(value, np.number) else value
    return number if math.isfinite(number) else None


def count_band(units: float) -> str:
    """
    Return the class of the count band a unit count falls in.

    Args:
        units (float):
            A unit count, summed or per record.

    Returns:
        str:
            The class of its band in :data:`COUNT_BANDS`.
    """
    for lower_bound, name in COUNT_BANDS:
        if units >= lower_bound:
            return name
    return 'RES1'


def unit_contribution(
    occupancy_class: str, resunits: object
) -> tuple[float | int | None, bool]:
    """
    Return the units one NSI record contributes to a count.

    Args:
        occupancy_class (str):
            The record's parsed class.
        resunits (object):
            The record's ``resunits``, or None where it has none.

    Returns:
        tuple[float or int or None, bool]:
            For a housing class (:data:`UNIT_COUNT_CLASSES`), its ``resunits`` where
            that is a positive number, otherwise the class midpoint, and whether the
            midpoint was used. For any other class, None and False.
    """
    if occupancy_class not in UNIT_COUNT_CLASSES:
        return None, False
    stated = stated_number(resunits)
    if stated is not None and stated > 0:
        return stated, False
    return UNIT_MIDPOINTS[occupancy_class], True


def point_class(occupancy_class: str, units: float | None) -> str:
    """
    Return a record's class after the count rule.

    Args:
        occupancy_class (str):
            The record's parsed class.
        units (float or None):
            The units it contributes (:func:`unit_contribution`).

    Returns:
        str:
            The count band of ``units`` (:func:`count_band`) for a record of
            :data:`COUNTED_FAMILIES`, otherwise the class itself.
    """
    # RES3A to RES3F count as RES3
    family = 'RES3' if occupancy_class in RES3_CLASSES else occupancy_class
    if family in COUNTED_FAMILIES and units is not None:
        return count_band(units)
    return occupancy_class
