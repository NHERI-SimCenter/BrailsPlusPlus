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


"""Tests for the Hazus earthquake height class inferer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import brails.inferers
from brails.inferers.hazus_inferer_earthquake.hazus_eq_diagnosis import (
    DIAGNOSIS_OK,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_height_class_inferer import (
    HazusEQHeightClassInferer,
)
from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    get_hazus_height_classes_by_type,
)
from brails.utils import Importer

if TYPE_CHECKING:
    from collections.abc import Callable

    from brails.types.asset_inventory import AssetInventory

# Types the damage models treat without a height class.
UNCLASSED_TYPES = ('W1', 'W2', 'S3', 'PC1', 'MH')


def _classify(
    build_inventory: Callable[..., AssetInventory],
    structure_type: str,
    stories: object,
) -> object:
    """
    Classify one building.

    Args:
        build_inventory (Callable[..., AssetInventory]):
            The inventory factory fixture.
        structure_type (str):
            The drawn structure type.
        stories (object):
            The story count.

    Returns:
        object:
            The height class assigned.
    """
    inventory = build_inventory(
        {'a': {'StructureType': structure_type, 'NumberOfStories': stories}}
    )
    return HazusEQHeightClassInferer().draw(inventory)['a']


@pytest.mark.parametrize(
    ('structure_type', 'stories', 'expected'),
    [
        ('C1', 1, 'Low-Rise'),
        ('C1', 3, 'Low-Rise'),
        ('C1', 4, 'Mid-Rise'),
        ('C1', 7, 'Mid-Rise'),
        ('C1', 8, 'High-Rise'),
        ('C1', 60, 'High-Rise'),
        ('URM', 1, 'Low-Rise'),
        ('URM', 2, 'Low-Rise'),
        ('URM', 3, 'Mid-Rise'),
        ('URM', 40, 'Mid-Rise'),
        ('RM1', 3, 'Low-Rise'),
        ('RM1', 4, 'Mid-Rise'),
        ('RM1', 30, 'Mid-Rise'),
    ],
)
def test_the_table_five_one_bands(
    build_inventory: Callable[..., AssetInventory],
    structure_type: str,
    stories: int,
    expected: str,
) -> None:
    """URM splits at two stories, RM1 has no High-Rise, the rest 3/4 and 7/8."""
    assert _classify(build_inventory, structure_type, stories) == expected


@pytest.mark.parametrize('structure_type', UNCLASSED_TYPES)
def test_types_without_a_height_class_are_written_null(
    build_inventory: Callable[..., AssetInventory],
    structure_type: str,
) -> None:
    """The damage models for these types ignore height."""
    assert _classify(build_inventory, structure_type, 5) is None


@pytest.mark.parametrize('structure_type', UNCLASSED_TYPES)
def test_a_type_without_a_height_class_needs_no_story_count(
    build_inventory: Callable[..., AssetInventory],
    structure_type: str,
) -> None:
    """Nothing is read that nothing is done with."""
    inventory = build_inventory({'a': {'StructureType': structure_type}})
    verdicts = HazusEQHeightClassInferer().check(inventory)
    assert verdicts.loc['a', 'diagnosis'] == DIAGNOSIS_OK
    assert HazusEQHeightClassInferer().draw(inventory)['a'] is None


def test_every_hazus_type_is_covered(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """No structure type the draw can produce is left unclassified."""
    classes = get_hazus_height_classes_by_type()
    for structure_type, bands in classes.items():
        result = _classify(build_inventory, structure_type, 5)
        if bands is None:
            assert result is None
        else:
            assert result in bands


def test_the_diagnosis_frame_carries_the_shared_columns(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A caller reads one shape whichever inferer it screened with."""
    inventory = build_inventory(
        {'a': {'StructureType': 'C1', 'NumberOfStories': 2}}
    )
    verdicts = HazusEQHeightClassInferer().check(inventory)
    assert list(verdicts.columns) == ['diagnosis', 'restriction', 'conflict']
    assert verdicts.loc['a', 'restriction'] == ''
    assert verdicts.loc['a', 'conflict'] == ''


@pytest.mark.parametrize(
    ('features', 'expected'),
    [
        ({'NumberOfStories': 2}, 'no StructureType'),
        ({'StructureType': 'W3', 'NumberOfStories': 2},
         'not a Hazus structure type'),
        ({'StructureType': 'C1'}, 'no NumberOfStories'),
        ({'StructureType': 'C1', 'NumberOfStories': 'two'}, 'not a number'),
        ({'StructureType': 'C1', 'NumberOfStories': 0}, 'one or more'),
        ({'StructureType': 'C1', 'NumberOfStories': 2.5}, 'whole number'),
    ],
)
def test_unusable_drivers_are_diagnosed(
    build_inventory: Callable[..., AssetInventory],
    features: dict,
    expected: str,
) -> None:
    """A missing or unusable driver produces a diagnosis of its own."""
    inventory = build_inventory({'a': features})
    verdicts = HazusEQHeightClassInferer().check(inventory)
    assert expected in verdicts.loc['a', 'diagnosis']


@pytest.mark.parametrize('key', ['StructureType', 'NumberOfStories'])
def test_a_per_world_driver_is_diagnosed(
    build_inventory: Callable[..., AssetInventory], key: str
) -> None:
    """Multi-world inventories are planned, but not supported yet."""
    features = {'StructureType': 'C1', 'NumberOfStories': 2}
    features[key] = [features[key], features[key]]
    inventory = build_inventory({'a': features}, n_pw=2)
    verdicts = HazusEQHeightClassInferer().check(inventory)
    assert 'per-world list' in verdicts.loc['a', 'diagnosis']


def test_infer_writes_the_height_class(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The class lands on a copy, leaving the input alone."""
    inventory = build_inventory(
        {'a': {'StructureType': 'URM', 'NumberOfStories': 3}}
    )
    output = HazusEQHeightClassInferer().infer(inventory)
    _, written = output.get_asset_features('a')
    _, original = inventory.get_asset_features('a')
    assert written['HeightClass'] == 'Mid-Rise'
    assert 'HeightClass' not in original


def test_infer_writes_a_null_rather_than_nothing(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A null height class is a value the damage library expects to see."""
    inventory = build_inventory(
        {'a': {'StructureType': 'W1', 'NumberOfStories': 2}}
    )
    _, written = HazusEQHeightClassInferer().infer(
        inventory
    ).get_asset_features('a')
    assert 'HeightClass' in written
    assert written['HeightClass'] is None


def test_infer_names_the_assets_it_cannot_classify(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Failing loudly is the point; a silent skip would look complete."""
    inventory = build_inventory(
        {
            'good': {'StructureType': 'C1', 'NumberOfStories': 2},
            'bad': {'NumberOfStories': 2},
        }
    )
    with pytest.raises(ValueError, match='bad'):
        HazusEQHeightClassInferer().infer(inventory)


def test_draw_refuses_an_asset_that_check_would_reject(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """draw() assumes a screened inventory and says so when it is not."""
    inventory = build_inventory({'a': {'StructureType': 'C1'}})
    with pytest.raises(ValueError, match='check'):
        HazusEQHeightClassInferer().draw(inventory)


def test_an_empty_inventory_is_not_an_error(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A caller that filtered everything out gets empty results."""
    inventory = build_inventory({})
    inferer = HazusEQHeightClassInferer()
    assert inferer.check(inventory).empty
    assert inferer.draw(inventory).empty


def test_the_feature_names_are_configurable(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A caller with its own vocabulary is not forced into ours."""
    inventory = build_inventory({'a': {'Type': 'C1', 'Stories': 9}})
    inferer = HazusEQHeightClassInferer(
        structure_type_key='Type',
        stories_key='Stories',
        height_class_key='Height',
    )
    _, written = inferer.infer(inventory).get_asset_features('a')
    assert written['Height'] == 'High-Rise'


def test_the_importer_finds_the_inferer() -> None:
    """The class is reachable the way the workflows reach classes."""
    assert (
        Importer().get_class('HazusEQHeightClassInferer')
        is HazusEQHeightClassInferer
    )
    assert hasattr(brails.inferers, 'HazusEQHeightClassInferer')
