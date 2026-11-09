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
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.

"""Tests for the Hazus EQ design level inferer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import brails.inferers
from brails.inferers.hazus_inferer_earthquake.hazus_eq_design_level_inferer import (
    HazusEQDesignLevelInferer,
    apply_w1_carveout,
    assign_design_levels,
)
from brails.utils import Importer

if TYPE_CHECKING:
    from collections.abc import Callable

    from brails.types.asset_inventory import AssetInventory

# ---------------------------------------------------------------------
# assign_design_levels


def test_california_zone_4_boundaries() -> None:
    """Pin the Table C-10 zone 4 era breaks at 1940/1941 and 1975/1976."""
    years = pd.Series(
        {'a': 1800, 'b': 1940, 'c': 1941, 'd': 1975, 'e': 1976, 'f': 2026}
    )
    levels = assign_design_levels(years, state='California', seismic_zone=4)
    assert levels.tolist() == [
        'Pre-Code',
        'Pre-Code',
        'Moderate-Code',
        'Moderate-Code',
        'High-Code',
        'High-Code',
    ]
    assert levels.index.equals(years.index)
    assert levels.name == 'DesignLevel'


def test_california_zone_3_caps_at_moderate() -> None:
    """The Table C-10 zone 3 row never rises above Moderate-Code."""
    years = pd.Series({'a': 1940, 'b': 1950, 'c': 1976, 'd': 2000})
    levels = assign_design_levels(years, state='California', seismic_zone=3)
    assert levels.tolist() == [
        'Pre-Code',
        'Moderate-Code',
        'Moderate-Code',
        'Moderate-Code',
    ]


def test_generic_zone_4_boundaries() -> None:
    """Pin the generic zone 4 era breaks at 1975/1976 and 1995/1996."""
    years = pd.Series({'a': 1975, 'b': 1976, 'c': 1995, 'd': 1996, 'e': 2026})
    levels = assign_design_levels(years, state='Texas', seismic_zone='4')
    assert levels.tolist() == [
        'Pre-Code',
        'Moderate-Code',
        'Moderate-Code',
        'High-Code',
        'High-Code',
    ]


@pytest.mark.parametrize(
    ('zone', 'expected'),
    [
        ('3', ['Pre-Code', 'Moderate-Code', 'Moderate-Code']),
        ('2B', ['Pre-Code', 'Low-Code', 'Moderate-Code']),
        ('2A', ['Pre-Code', 'Low-Code', 'Low-Code']),
        ('0', ['Pre-Code', 'Pre-Code', 'Low-Code']),
    ],
)
def test_generic_zone_rows(zone: str, expected: list[str]) -> None:
    """Probe one year in each era of every generic zone row."""
    years = pd.Series({'a': 1950, 'b': 1980, 'c': 2000})
    levels = assign_design_levels(years, state='Texas', seismic_zone=zone)
    assert levels.tolist() == expected


def test_zone_spelling_is_normalized() -> None:
    """A seismic zone resolves the same way in either case and as an integer."""
    years = pd.Series({'a': 1980})
    assert assign_design_levels(years, 'Texas', '2b').tolist() == ['Low-Code']
    assert assign_design_levels(years, 'Texas', '2B').tolist() == ['Low-Code']
    assert assign_design_levels(years, 'California', 4).tolist() == ['High-Code']


def test_a_two_letter_state_code_is_rejected() -> None:
    """A two-letter state code raises and asks for the full state name."""
    years = pd.Series({'a': 1980})
    with pytest.raises(ValueError, match='Give the full name'):
        assign_design_levels(years, state='CA', seismic_zone=4)


@pytest.mark.parametrize(
    ('state', 'zone'),
    [
        ('California', '2B'),
        ('Texas', '5'),
    ],
)
def test_unknown_zone_is_rejected(state: str, zone: str) -> None:
    """A zone outside the selected state table raises with the valid zones."""
    years = pd.Series({'a': 1980})
    with pytest.raises(ValueError, match='Valid zones'):
        assign_design_levels(years, state=state, seismic_zone=zone)


@pytest.mark.parametrize(
    'bad_year', [None, float('nan'), 0, -1, 1975.5, 'old']
)
def test_invalid_years_are_rejected(bad_year: object) -> None:
    """Missing and malformed years raise and name the offending asset."""
    years = pd.Series({'good': 1980, 'bad': bad_year})
    with pytest.raises(ValueError, match='bad'):
        assign_design_levels(years, state='California', seismic_zone=4)


def test_integral_float_and_string_years_pass() -> None:
    """Whole years stored as floats or digit strings are accepted."""
    years = pd.Series({'a': 1975.0, 'b': '1976'})
    levels = assign_design_levels(years, state='California', seismic_zone=4)
    assert levels.tolist() == ['Moderate-Code', 'High-Code']


# ---------------------------------------------------------------------
# the W1 carve-out


def test_w1_carveout() -> None:
    """Pre-Code W1 rises to Moderate-Code and everything else stays."""
    levels = pd.Series(
        {
            'a': 'Pre-Code',
            'b': 'Pre-Code',
            'c': 'Moderate-Code',
            'd': 'High-Code',
        }
    )
    types = pd.Series({'a': 'W1', 'b': 'URM', 'c': 'W1', 'd': 'W1'})
    adjusted = apply_w1_carveout(levels, types)
    assert adjusted.tolist() == [
        'Moderate-Code',
        'Pre-Code',
        'Moderate-Code',
        'High-Code',
    ]
    assert levels['a'] == 'Pre-Code'


def test_w1_carveout_rejects_mismatched_index() -> None:
    """Series over different assets cannot be combined."""
    levels = pd.Series({'a': 'Pre-Code'})
    types = pd.Series({'b': 'W1'})
    with pytest.raises(ValueError, match='same index'):
        apply_w1_carveout(levels, types)


# ---------------------------------------------------------------------
# the inferer class


def test_infer_assigns_every_asset(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The inferer writes a design level to each asset and keeps the input."""
    inventory = build_inventory(
        {
            'a': {'YearBuilt': 1930},
            'b': {'YearBuilt': 1950},
            'c': {'YearBuilt': 1990},
        }
    )
    inferer = HazusEQDesignLevelInferer(state='California', seismic_zone=4)
    result = inferer.infer(inventory)
    expected = {'a': 'Pre-Code', 'b': 'Moderate-Code', 'c': 'High-Code'}
    for asset_id, level in expected.items():
        _, features = result.get_asset_features(asset_id)
        assert features['DesignLevel'] == level
        _, original = inventory.get_asset_features(asset_id)
        assert 'DesignLevel' not in original


def test_infer_honors_custom_keys(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The year is read and the level written under configured names."""
    inventory = build_inventory({'a': {'erected': 1950}})
    inferer = HazusEQDesignLevelInferer(
        year_built_key='erected', design_level_key='dl'
    )
    _, features = inferer.infer(inventory).get_asset_features('a')
    assert features['dl'] == 'Moderate-Code'


def test_infer_reports_the_configured_year_key(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The invalid-year error names the configured feature name."""
    inventory = build_inventory({'a': {'erected': 'unknown'}})
    inferer = HazusEQDesignLevelInferer(year_built_key='erected')
    with pytest.raises(ValueError, match='erected must be a positive'):
        inferer.infer(inventory)


def test_infer_requires_year_built(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """An asset without a construction year fails loudly by id."""
    inventory = build_inventory(
        {'a': {'YearBuilt': 1950}, 'b': {'OccupancyClass': 'RES1'}}
    )
    with pytest.raises(ValueError, match='Missing on: b'):
        HazusEQDesignLevelInferer().infer(inventory)


def test_infer_rejects_per_world_years(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Per-world year lists are refused until multi-world support lands."""
    inventory = build_inventory({'a': {'YearBuilt': [1950, 1990]}}, n_pw=2)
    with pytest.raises(ValueError, match='not supported yet'):
        HazusEQDesignLevelInferer().infer(inventory)


def test_v61_method_is_reserved() -> None:
    """Requesting the v6.1 method fails until it is implemented."""
    with pytest.raises(NotImplementedError):
        HazusEQDesignLevelInferer(method='6.1')


def test_unknown_method_is_rejected() -> None:
    """A method outside 5.1 and 6.1 raises a value error."""
    with pytest.raises(ValueError, match='method'):
        HazusEQDesignLevelInferer(method='7.0')


def test_bad_zone_fails_at_construction() -> None:
    """Configuration errors surface when the inferer is built."""
    with pytest.raises(ValueError, match='Valid zones'):
        HazusEQDesignLevelInferer(state='California', seismic_zone='2B')


def test_importer_finds_the_inferer() -> None:
    """The class is discoverable through the Importer by its unique name."""
    importer = Importer()
    assert importer.get_class(
        'HazusEQDesignLevelInferer'
    ) is HazusEQDesignLevelInferer


def test_package_level_import() -> None:
    """The class is importable directly from brails.inferers."""
    assert brails.inferers.HazusEQDesignLevelInferer is HazusEQDesignLevelInferer
