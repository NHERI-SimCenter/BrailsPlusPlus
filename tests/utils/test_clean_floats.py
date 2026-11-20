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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.


"""Tests for clean_floats, which prepares a payload for strict JSON."""

from __future__ import annotations

import json
import math

import pytest

from brails.utils.clean_floats import clean_floats


def test_integral_float_becomes_int() -> None:
    """A whole-number float becomes an int and other floats are kept."""
    assert clean_floats({'a': 2.0, 'b': [2.5, -3.0]}) == {'a': 2, 'b': [2.5, -3]}
    assert type(clean_floats(2.0)) is int


def test_nan_becomes_none_and_is_written_as_null() -> None:
    """A NaN becomes None, which json.dumps writes as null."""
    cleaned = clean_floats({'a': math.nan, 'b': [1.5, math.nan]})
    assert cleaned == {'a': None, 'b': [1.5, None]}
    assert json.dumps(cleaned, allow_nan=False) == '{"a": null, "b": [1.5, null]}'


def test_tuple_is_returned_as_it_is() -> None:
    """A tuple is not traversed, so the floats inside it are kept."""
    pair = (1.0, math.nan, math.inf)
    assert clean_floats({'coordinates': pair})['coordinates'] is pair


@pytest.mark.parametrize('value', [math.inf, -math.inf])
def test_infinite_value_raises(value: float) -> None:
    """An infinite value raises, also inside a list."""
    with pytest.raises(ValueError, match='infinite'):
        clean_floats({'a': [0.0, value]})
