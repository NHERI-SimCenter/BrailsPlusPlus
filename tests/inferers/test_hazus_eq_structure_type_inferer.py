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


"""Tests for the Hazus earthquake structure type inferer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import brails.inferers
from brails.inferers.hazus_inferer_earthquake.hazus_eq_diagnosis import (
    RESTRICTION_BUILDING_TYPE,
    RESTRICTION_DESIGN_LEVEL,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_structure_type_inferer import (
    UNREINFORCED_MASONRY_TYPES,
    HazusEQStructureTypeInferer,
)
from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    get_hazus_general_type_membership,
    get_hazus_occ_type_mapping,
)
from brails.inferers.inference_diagnosis import DIAGNOSIS_OK
from brails.utils import Importer
from brails.utils.deterministic_sampling import asset_key_digest

if TYPE_CHECKING:
    from collections.abc import Callable

    from brails.types.asset_inventory import AssetInventory

# A low-rise retail building.
STORE = {
    'OccupancyClass': 'COM1',
    'NumberOfStories': 2,
    'YearBuilt': 1960,
    'DesignLevel': 'Moderate-Code',
}

# A mid-rise cell whose only masonry type is URM, so a design level above
# Low-Code leaves a masonry-tagged building with nothing to draw.
MASONRY_WALKUP = {
    'OccupancyClass': 'COM8',
    'NumberOfStories': 5,
    'YearBuilt': 1945,
    'DesignLevel': 'Moderate-Code',
    'BuildingType': 'M',
}

# A high-rise, where the tables carry no wood at all.
WOOD_TOWER = {
    'OccupancyClass': 'RES3',
    'NumberOfStories': 9,
    'YearBuilt': 1960,
    'DesignLevel': 'Moderate-Code',
    'BuildingType': 'W',
}

HOME = {
    'OccupancyClass': 'RES1',
    'YearBuilt': 1930,
    'DesignLevel': 'Pre-Code',
}


def _inferer(**kwargs: object) -> HazusEQStructureTypeInferer:
    """
    Build an inferer with a fixed seed unless the test says otherwise.

    Args:
        **kwargs (object):
            Constructor arguments to override.

    Returns:
        HazusEQStructureTypeInferer:
            The inferer.
    """
    settings = {'state': 'California', 'seed': 20260828}
    settings.update(kwargs)
    return HazusEQStructureTypeInferer(**settings)


def _spread(
    build_inventory: Callable[..., AssetInventory],
    features: dict,
    count: int = 400,
    **kwargs: object,
) -> list[str]:
    """
    Draw for many assets that share one set of features.

    Args:
        build_inventory (Callable[..., AssetInventory]):
            The inventory factory fixture.
        features (dict):
            Features every asset carries.
        count (int):
            How many assets to draw for.
        **kwargs (object):
            Constructor arguments to override.

    Returns:
        list[str]:
            The types drawn.
    """
    inventory = build_inventory(
        {f'asset_{index}': dict(features) for index in range(count)}
    )
    return _inferer(**kwargs).draw(inventory).tolist()


# ---------------------------------------------------------------------
# configuration


def test_a_full_state_name_is_required() -> None:
    """A two-letter code would select the wrong table set silently."""
    with pytest.raises(ValueError, match='two-letter codes'):
        HazusEQStructureTypeInferer(state='CA')


def test_an_unknown_state_is_refused() -> None:
    """A misspelled state must not fall through to a default region."""
    with pytest.raises(ValueError, match='not a state'):
        HazusEQStructureTypeInferer(state='Califrnia')


def test_each_inferer_draws_its_own_seed() -> None:
    """A probabilistic result should look probabilistic between runs."""
    first = HazusEQStructureTypeInferer(state='California')
    second = HazusEQStructureTypeInferer(state='California')
    assert first.seed != second.seed


def test_a_recorded_seed_repeats_the_run(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The seed a run drew is what makes that run repeatable."""
    inventory = build_inventory(
        {f'asset_{index}': dict(STORE) for index in range(50)}
    )
    inferer = HazusEQStructureTypeInferer(state='California')
    drawn = inferer.draw(inventory)
    repeat = HazusEQStructureTypeInferer(state='California', seed=inferer.seed)
    assert repeat.draw(inventory).equals(drawn)


def test_the_state_selects_the_region() -> None:
    """Table A-22 decides which table set an asset draws from."""
    assert _inferer(state='Iowa').region == 'Mid-West'
    assert _inferer(state='California').region == 'West Coast'


# ---------------------------------------------------------------------
# the draw


def test_a_type_is_drawn_from_the_cell(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """An ordinary asset draws one of its cell's types."""
    type_lists, _ = get_hazus_occ_type_mapping()
    drawn = set(_spread(build_inventory, STORE))
    assert drawn
    assert drawn <= set(type_lists['West Coast']['low_rise'])


def test_frequencies_follow_the_table(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The draw reproduces the manual's percentages, not just its types."""
    sample = 4000
    type_lists, type_weights = get_hazus_occ_type_mapping()
    row = type_weights['West Coast']['COM1']['low_rise']['pre_1950']
    expected = dict(zip(type_lists['West Coast']['low_rise'], row))
    drawn = _spread(
        build_inventory,
        dict(STORE, YearBuilt=1930, DesignLevel='Pre-Code'),
        count=sample,
    )
    for structure_type, weight in expected.items():
        share = drawn.count(structure_type) / len(drawn)
        target = weight / sum(row)
        if target == 0:
            assert share == 0
            continue
        tolerance = 4 * np.sqrt(target * (1 - target) / sample)
        assert share == pytest.approx(target, abs=tolerance)


def test_a_single_family_home_needs_no_story_count(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """RES1 is tabulated by state and era, with no height dimension."""
    inventory = build_inventory({'a': dict(HOME)})
    assert _inferer().check(inventory).loc['a', 'diagnosis'] == DIAGNOSIS_OK
    assert _inferer().draw(inventory)['a'] in {'W1', 'S3', 'S5', 'C2', 'RM1',
                                               'URM'}


def test_a_home_outside_the_west_coast_needs_no_year(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Only the West Coast tables are stratified by era."""
    inventory = build_inventory(
        {'a': {'OccupancyClass': 'RES1', 'DesignLevel': 'Pre-Code'}}
    )
    inferer = _inferer(state='Iowa')
    assert inferer.check(inventory).loc['a', 'diagnosis'] == DIAGNOSIS_OK
    assert inferer.draw(inventory)['a'] in {'W1', 'C2', 'URM'}


def test_multifamily_subtypes_draw_as_res3(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The earthquake tables have one RES3 row for all six subtypes."""
    subtype = dict(STORE, OccupancyClass='RES3B')
    parent = dict(STORE, OccupancyClass='RES3')
    subtype_draw = _inferer().draw(build_inventory({'a': subtype}))
    parent_draw = _inferer().draw(build_inventory({'a': parent}))
    assert subtype_draw['a'] == parent_draw['a']


def test_a_subtype_keeps_its_occupancy_class(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Folding is for the table lookup; the asset keeps what it carried."""
    inventory = build_inventory({'a': dict(STORE, OccupancyClass='RES3B')})
    output = _inferer().infer(inventory)
    _, features = output.get_asset_features('a')
    assert features['OccupancyClass'] == 'RES3B'


# ---------------------------------------------------------------------
# restrictions


def test_an_observed_building_type_restricts_the_draw(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A concrete building cannot come out of the draw as wood."""
    members = set(get_hazus_general_type_membership()['C'])
    drawn = set(_spread(build_inventory, dict(STORE, BuildingType='C')))
    assert drawn
    assert drawn <= members


def test_the_building_type_can_be_ignored(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A caller who distrusts the observed type can turn it off."""
    members = set(get_hazus_general_type_membership()['C'])
    drawn = set(
        _spread(
            build_inventory,
            dict(STORE, BuildingType='C'),
            use_building_type=False,
        )
    )
    assert not drawn <= members


def test_a_design_level_above_low_code_removes_the_masonry_infill_types(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """S5, C3 and URM have no archetype above Low-Code."""
    drawn = set(
        _spread(
            build_inventory,
            dict(STORE, YearBuilt=1945, DesignLevel='Moderate-Code'),
            count=4000,
        )
    )
    assert not drawn & set(UNREINFORCED_MASONRY_TYPES)


def test_pre_code_keeps_the_masonry_infill_types(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The same cell keeps them where the design level allows them."""
    drawn = set(
        _spread(
            build_inventory,
            dict(STORE, YearBuilt=1930, DesignLevel='Pre-Code'),
            count=4000,
        )
    )
    assert drawn & set(UNREINFORCED_MASONRY_TYPES)


def test_a_restriction_down_to_one_type_draws_it_every_time(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Weights are renormalized after a restriction, not left short."""
    drawn = set(
        _spread(build_inventory, dict(HOME, BuildingType='W'), count=50)
    )
    assert drawn == {'W1'}


# ---------------------------------------------------------------------
# diagnoses


def test_a_masonry_walkup_above_low_code_is_diagnosed(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The cell holds URM as its only masonry, and the level removes it."""
    inventory = build_inventory({'a': dict(MASONRY_WALKUP)})
    row = _inferer().check(inventory).loc['a']
    assert row['restriction'] == RESTRICTION_DESIGN_LEVEL
    assert row['conflict'] == 'BuildingType|YearBuilt'


def test_a_type_absent_only_from_this_era_names_the_year(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Where another era carries the class, the story count cannot help."""
    inventory = build_inventory(
        {
            'a': {
                'OccupancyClass': 'IND4',
                'NumberOfStories': 2,
                'YearBuilt': 1960,
                'DesignLevel': 'Moderate-Code',
                'BuildingType': 'W',
            }
        }
    )
    row = _inferer().check(inventory).loc['a']
    assert row['restriction'] == RESTRICTION_BUILDING_TYPE
    assert row['conflict'] == 'BuildingType|YearBuilt'


def test_a_wood_high_rise_is_diagnosed(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The tables carry no wood above seven stories."""
    inventory = build_inventory({'a': dict(WOOD_TOWER)})
    row = _inferer().check(inventory).loc['a']
    assert row['restriction'] == RESTRICTION_BUILDING_TYPE
    assert row['conflict'] == 'BuildingType|NumberOfStories'


def test_a_restricted_home_names_its_occupancy(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """RES1 has no story band, so the pair is the occupancy instead."""
    inventory = build_inventory({'a': dict(HOME, BuildingType='S')})
    row = _inferer().check(inventory).loc['a']
    assert row['restriction'] == RESTRICTION_BUILDING_TYPE
    assert row['conflict'] == 'BuildingType|OccupancyClass'


def test_a_cell_the_tables_leave_empty_is_diagnosed(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """An eight-story retail building has no row, before any restriction."""
    inventory = build_inventory({'a': dict(STORE, NumberOfStories=8)})
    row = _inferer().check(inventory).loc['a']
    assert 'no structure type' in row['diagnosis']
    assert row['restriction'] == ''
    assert row['conflict'] == ''


def test_a_state_without_a_home_row_is_diagnosed(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Guam is a West Coast state with no RES1 row in the tables."""
    inventory = build_inventory({'a': dict(HOME)})
    row = _inferer(state='Guam').check(inventory).loc['a']
    assert 'has no RES1 row' in row['diagnosis']


@pytest.mark.parametrize(
    ('features', 'expected'),
    [
        ({'NumberOfStories': 2, 'YearBuilt': 1960,
          'DesignLevel': 'Moderate-Code'}, 'no OccupancyClass'),
        (dict(STORE, OccupancyClass='CAFE'), 'has no row'),
        ({'OccupancyClass': 'COM1', 'YearBuilt': 1960,
          'DesignLevel': 'Moderate-Code'}, 'no NumberOfStories'),
        (dict(STORE, NumberOfStories='two'), 'not a number'),
        (dict(STORE, NumberOfStories=0), 'one or more'),
        (dict(STORE, NumberOfStories=2.5), 'whole number'),
        ({'OccupancyClass': 'COM1', 'NumberOfStories': 2,
          'DesignLevel': 'Moderate-Code'}, 'no YearBuilt'),
        (dict(STORE, YearBuilt=2050), 'sits outside'),
        (dict(STORE, YearBuilt='old'), 'not a number'),
        ({'OccupancyClass': 'COM1', 'NumberOfStories': 2,
          'YearBuilt': 1960}, 'no DesignLevel'),
        (dict(STORE, DesignLevel='HighCode'), 'not a Hazus design level'),
        (dict(STORE, BuildingType='X'), 'not a Hazus general building type'),
    ],
)
def test_unusable_drivers_are_diagnosed(
    build_inventory: Callable[..., AssetInventory],
    features: dict,
    expected: str,
) -> None:
    """A driver that is present but unusable is named, not coerced."""
    inventory = build_inventory({'a': features})
    assert expected in _inferer().check(inventory).loc['a', 'diagnosis']


@pytest.mark.parametrize(
    'key', ['OccupancyClass', 'NumberOfStories', 'YearBuilt', 'DesignLevel',
            'BuildingType']
)
def test_a_per_world_driver_is_diagnosed(
    build_inventory: Callable[..., AssetInventory], key: str
) -> None:
    """Multi-world inventories are planned, but currently not supported."""
    features = dict(STORE, BuildingType='C')
    features[key] = [features[key], features[key]]
    inventory = build_inventory({'a': features}, n_pw=2)
    assert 'per-world list' in _inferer().check(inventory).loc['a',
                                                               'diagnosis']


def _weight_rows(cell: object) -> list[list[float]]:
    """
    Flatten a table cell into the weight rows it holds.

    Args:
        cell (object):
            A weight row, or a dict of them nested by era or state.

    Returns:
        list[list[float]]:
            Every weight row the cell contains.
    """
    if isinstance(cell, dict):
        rows = []
        for nested in cell.values():
            rows.extend(_weight_rows(nested))
        return rows
    return [cell]


def test_the_tables_hold_no_cell_of_masonry_infill_alone() -> None:
    """
    Pin what makes one conflict shape unreachable.

    A design level can only empty a cell that an observed building type
    already narrowed, because no cell in the shipped tables is made up
    of S5, C3 and URM alone. The inferer handles the case anyway; this
    is what would change if a later edition of the manual changed the tables.
    """
    type_lists, type_weights = get_hazus_occ_type_mapping()
    checked = 0
    for region, occupancies in type_weights.items():
        for occupancy, bands in occupancies.items():
            for band, cell in bands.items():
                key = 'RES1' if occupancy == 'RES1' else band
                for row in _weight_rows(cell):
                    if not row or sum(row) <= 0:
                        continue
                    weights = dict(zip(type_lists[region][key], row))
                    infill = sum(
                        weight
                        for structure_type, weight in weights.items()
                        if structure_type in UNREINFORCED_MASONRY_TYPES
                    )
                    assert infill < sum(row)
                    checked += 1
    # Guards a vacuous pass: a flattener that returned nothing would
    # satisfy every assertion above it.
    assert checked > 0


# ---------------------------------------------------------------------
# band boundaries


def test_the_low_rise_band_ends_at_three_stories(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Wood is tabulated up to three stories and not above."""
    inventory = build_inventory(
        {
            'low': dict(STORE, NumberOfStories=3, BuildingType='W'),
            'mid': dict(STORE, NumberOfStories=4, BuildingType='W'),
        }
    )
    verdicts = _inferer().check(inventory)
    assert verdicts.loc['low', 'diagnosis'] == DIAGNOSIS_OK
    assert verdicts.loc['mid', 'restriction'] == RESTRICTION_BUILDING_TYPE


def test_the_mid_rise_band_ends_at_seven_stories(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """
    Retail is tabulated to seven stories and no further.

    COM1 is retail trade, which the manual does not carry above the
    mid-rise band. COM4, the office class, does carry a high-rise row,
    so the eighth story is the band boundary and not a gap in the
    tables.
    """
    inventory = build_inventory(
        {
            'mid': dict(STORE, NumberOfStories=7),
            'high': dict(STORE, NumberOfStories=8),
            'office': dict(STORE, OccupancyClass='COM4', NumberOfStories=8),
        }
    )
    verdicts = _inferer().check(inventory)
    assert verdicts.loc['mid', 'diagnosis'] == DIAGNOSIS_OK
    assert 'at 8 stories' in verdicts.loc['high', 'diagnosis']
    assert verdicts.loc['office', 'diagnosis'] == DIAGNOSIS_OK


# ---------------------------------------------------------------------
# determinism


def test_draws_survive_reordering(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """An asset's type belongs to the asset, not to its row number."""
    features = {name: dict(STORE) for name in ('a', 'b', 'c', 'd')}
    forward = _inferer().draw(build_inventory(features))
    backward = _inferer().draw(
        build_inventory({name: features[name] for name in reversed(list(features))})
    )
    assert forward.to_dict() == backward.to_dict()


def test_draws_survive_subsetting(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Debugging one asset must not change the type it draws."""
    features = {f'asset_{index}': dict(STORE) for index in range(20)}
    whole = _inferer().draw(build_inventory(features))
    part = _inferer().draw(
        build_inventory({'asset_7': features['asset_7']})
    )
    assert part['asset_7'] == whole['asset_7']


def test_two_seeds_draw_differently(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The seed reaches the draw."""
    features = {f'asset_{index}': dict(STORE) for index in range(100)}
    inventory = build_inventory(features)
    first = _inferer(seed=1).draw(inventory)
    second = _inferer(seed=2).draw(inventory)
    assert not first.equals(second)


def test_iterations_draw_differently(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A redraw asks the same asset for a fresh type, and repeats."""
    features = {f'asset_{index}': dict(STORE) for index in range(100)}
    inventory = build_inventory(features)
    inferer = _inferer()
    first = inferer.draw(inventory, iteration=0)
    second = inferer.draw(inventory, iteration=1)
    assert not first.equals(second)
    assert second.equals(inferer.draw(inventory, iteration=1))


def test_pinned_draw(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """One value fixed in place."""
    inventory = build_inventory({'asset_0': dict(STORE)})
    assert _inferer(seed=20260828).draw(inventory)['asset_0'] == 'W2'


def test_keys_may_be_passed_in(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """
    Digesting dominates a redraw, so a caller may digest once.

    Both draws come from one inferer, so the seed is the same and the
    keys are the only difference between them.
    """
    features = {f'asset_{index}': dict(STORE) for index in range(10)}
    inventory = build_inventory(features)
    keys = asset_key_digest(inventory.get_asset_ids())
    inferer = _inferer()
    assert inferer.draw(inventory, asset_keys=keys).equals(
        inferer.draw(inventory)
    )


def test_mismatched_keys_are_refused(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Keys that do not line up would draw for the wrong assets."""
    inventory = build_inventory(
        {f'asset_{index}': dict(STORE) for index in range(3)}
    )
    with pytest.raises(ValueError, match='one key per asset'):
        _inferer().draw(inventory, asset_keys=np.zeros(2, dtype=np.uint64))


def test_keys_of_the_wrong_kind_are_refused(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A float array would be cast to keys and quietly draw from those."""
    inventory = build_inventory({'a': dict(STORE)})
    with pytest.raises(ValueError, match='asset_key_digest'):
        _inferer().draw(inventory, asset_keys=np.array([0.5]))


def test_the_world_reaches_the_draw(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Worlds draw independently, which is what item by item means here."""
    inventory = build_inventory(
        {f'asset_{index}': dict(STORE) for index in range(100)}
    )
    first = _inferer(world=0).draw(inventory)
    second = _inferer(world=1).draw(inventory)
    assert not first.equals(second)


# ---------------------------------------------------------------------
# infer


def test_infer_writes_the_type(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The drawn type lands on a copy, leaving the input alone."""
    inventory = build_inventory({'a': dict(STORE)})
    output = _inferer().infer(inventory)
    _, written = output.get_asset_features('a')
    _, original = inventory.get_asset_features('a')
    assert written['StructureType'] in set(
        get_hazus_occ_type_mapping()[0]['West Coast']['low_rise']
    )
    assert 'StructureType' not in original


def test_infer_names_the_assets_it_cannot_draw_for(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Failing loudly is the point; a silent skip would look complete."""
    inventory = build_inventory(
        {'good': dict(STORE), 'bad': dict(WOOD_TOWER)}
    )
    with pytest.raises(ValueError, match='bad'):
        _inferer().infer(inventory)


def test_infer_previews_ten_failures_and_counts_the_rest(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """An inventory-wide failure must not print an inventory-wide message."""
    inventory = build_inventory(
        {f'asset_{index}': dict(WOOD_TOWER) for index in range(12)}
    )
    with pytest.raises(ValueError, match='and 2 more'):
        _inferer().infer(inventory)


def test_draw_refuses_an_asset_that_check_would_reject(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """draw() assumes a screened inventory and says so when it is not."""
    inventory = build_inventory({'a': dict(WOOD_TOWER)})
    with pytest.raises(ValueError, match='check'):
        _inferer().draw(inventory)


def test_an_empty_inventory_is_not_an_error(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A caller that filtered everything out gets empty results."""
    inventory = build_inventory({})
    assert _inferer().check(inventory).empty
    assert _inferer().draw(inventory).empty


def test_the_importer_finds_the_inferer() -> None:
    """The class is reachable the way the workflows reach classes."""
    assert (
        Importer().get_class('HazusEQStructureTypeInferer')
        is HazusEQStructureTypeInferer
    )
    assert hasattr(brails.inferers, 'HazusEQStructureTypeInferer')
