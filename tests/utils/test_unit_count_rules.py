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

"""Tests for the unit-count rules."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from brails.utils.unit_count_rules import (
    count_band,
    stated_number,
    unit_contribution,
)


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (3, 3),
        (2.5, 2.5),
        (np.int64(4), 4),
        (np.float32(0.5), 0.5),
        (True, None),
        (math.nan, None),
        (math.inf, None),
        ('3', None),
        (None, None),
    ],
)
def test_stated_number(value: Any, expected: Any) -> None:
    """A finite int, float or NumPy number is stated, and nothing else is."""
    assert stated_number(value) == expected


@pytest.mark.parametrize(
    ('units', 'expected'),
    [
        (0, 'RES1'),
        (1.0, 'RES1'),
        (1.9, 'RES1'),
        (2, 'RES3A'),
        (2.9, 'RES3A'),
        (3, 'RES3B'),
        (4.9, 'RES3B'),
        (5, 'RES3C'),
        (9.9, 'RES3C'),
        (10, 'RES3D'),
        (19.9, 'RES3D'),
        (20, 'RES3E'),
        (49.9, 'RES3E'),
        (50, 'RES3F'),
        (400, 'RES3F'),
    ],
)
def test_count_band(units: float, expected: str) -> None:
    """A count falls in the band whose lower bound it has reached."""
    assert count_band(units) == expected


@pytest.mark.parametrize(
    ('occupancy', 'resunits', 'expected'),
    [
        ('RES3B', 4, (4, False)),
        ('RES3B', 0, (3.5, True)),
        ('RES3B', -2, (3.5, True)),
        ('RES3B', None, (3.5, True)),
        ('RES3B', 'four', (3.5, True)),
        ('RES1', 0, (1.0, True)),
        ('RES2', 0, (1.0, True)),
        ('RES3A', 0, (2.0, True)),
        ('RES3C', 0, (7.0, True)),
        ('RES3D', 0, (14.5, True)),
        ('RES3E', 0, (34.5, True)),
        ('RES3F', 0, (100.0, True)),
        ('COM1', 12, (None, False)),
    ],
)
def test_unit_contribution(occupancy: str, resunits: Any, expected: tuple) -> None:
    """A housing record counts its units or its class midpoint."""
    assert unit_contribution(occupancy, resunits) == expected
