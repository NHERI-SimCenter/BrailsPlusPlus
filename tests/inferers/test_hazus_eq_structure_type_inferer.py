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
import pandas as pd
import pytest

import brails.inferers
from brails.inferers.hazus_inferer_earthquake.hazus_eq_design_level_inferer import (
    CA_DESIGN_LEVELS,
    GENERIC_DESIGN_LEVELS,
    assign_design_levels,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_structure_type_inferer import (
    UNREINFORCED_MASONRY_TYPES,
    HazusEQStructureTypeInferer,
)
from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    get_hazus_general_type_membership,
    get_hazus_occ_type_mapping,
    get_hazus_unreinforced_design_levels,
)
from brails.inferers.hazus_inferer_earthquake.hazus_table_completion import (
    get_completed_west_coast_weights,
    get_west_coast_fill_routes,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INFEASIBLE,
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
)
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

# A five-story school built 1950 to 1970, whose West Coast cell the tables
# print no row for.
SCHOOL = dict(STORE, OccupancyClass='EDU1', NumberOfStories=5)


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


def test_the_west_coast_allows_the_unreinforced_types_at_pre_code_only(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """
    The Appendix A footnote is stricter on the West Coast than elsewhere.

    Low-Code admits URM, C3 and S5 in the Mid-West and on the East
    Coast, and does not on the West Coast. California cannot show it:
    its own design level table has no Low-Code era, so the fixture is a
    generic-zone West Coast state.
    """
    masonry = {
        'OccupancyClass': 'COM8',
        'NumberOfStories': 5,
        'YearBuilt': 1945,
        'DesignLevel': 'Low-Code',
        'BuildingType': 'M',
    }
    inventory = build_inventory({'a': dict(masonry)})
    west = _inferer(state='Washington').check(inventory).loc['a']
    assert west['diagnosis'] == DIAGNOSIS_INFEASIBLE
    midwest = _inferer(state='Iowa').check(inventory).loc['a']
    assert midwest['diagnosis'] == DIAGNOSIS_OK
    drawn = set(_spread(build_inventory, masonry, count=200, state='Iowa'))
    assert 'URM' in drawn


def test_the_legality_set_can_be_overridden(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A user who wants their own judgment applied is not locked out."""
    masonry = {
        'OccupancyClass': 'COM8',
        'NumberOfStories': 5,
        'YearBuilt': 1945,
        'DesignLevel': 'Low-Code',
        'BuildingType': 'M',
    }
    inventory = build_inventory({'a': dict(masonry)})
    strict = _inferer(state='Washington').check(inventory).loc['a']
    assert strict['diagnosis'] == DIAGNOSIS_INFEASIBLE
    relaxed = _inferer(
        state='Washington',
        unreinforced_design_levels=('Pre-Code', 'Low-Code'),
    )
    assert relaxed.check(inventory).loc['a', 'diagnosis'] == DIAGNOSIS_OK
    assert relaxed.draw(inventory)['a'] == 'URM'


def test_the_legality_sets_come_from_the_tables() -> None:
    """The inferer reads the footnote rather than restating it."""
    levels = get_hazus_unreinforced_design_levels()
    assert levels['West Coast'] == ('Pre-Code',)
    assert set(levels['Mid-West']) == {'Pre-Code', 'Low-Code'}
    assert set(levels['East Coast']) == {'Pre-Code', 'Low-Code'}


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
    assert row['diagnosis'] == DIAGNOSIS_INFEASIBLE
    assert row['resolvable_by'] == (
        'OccupancyClass', 'NumberOfStories', 'YearBuilt', 'BuildingType'
    )


def test_an_occupancy_the_tables_do_not_know_is_invalid(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A lookup that cannot run is not a lookup that returned nothing."""
    inventory = build_inventory(
        {
            'unknown': dict(STORE, OccupancyClass='CAFE'),
            'subtype_typo': dict(STORE, OccupancyClass='RES3Z'),
        }
    )
    verdicts = _inferer().check(inventory)
    assert all(verdicts['diagnosis'] == DIAGNOSIS_INVALID)
    assert all(verdicts['resolvable_by'] == ())


def test_a_legality_set_the_year_cannot_reach_is_refused() -> None:
    """
    The year is offered on the strength of the earliest design level.

    A set that leaves it out would make that offer false, so the
    constructor refuses one rather than reporting years that resolve
    nothing.
    """
    with pytest.raises(ValueError, match='must admit Pre-Code'):
        _inferer(unreinforced_design_levels=('Low-Code',))
    with pytest.raises(ValueError, match='not Hazus design levels'):
        _inferer(unreinforced_design_levels=('PreCode',))


def test_nothing_is_offered_where_the_lookup_could_not_run(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A feature cannot be relaxed into an invalid or unsupported lookup."""
    inventory = build_inventory(
        {
            'no_driver': {'NumberOfStories': 2, 'YearBuilt': 1960,
                          'DesignLevel': 'Moderate-Code'},
            'unusable': dict(STORE, NumberOfStories=0),
            'unsupported': dict(STORE, OccupancyClass=['COM1', 'COM1']),
        },
        n_pw=2,
    )
    verdicts = _inferer().check(inventory)
    assert verdicts.loc['no_driver', 'diagnosis'] == DIAGNOSIS_INVALID
    assert verdicts.loc['unusable', 'diagnosis'] == DIAGNOSIS_INVALID
    assert verdicts.loc['unsupported', 'diagnosis'] == DIAGNOSIS_UNSUPPORTED
    assert all(verdicts['resolvable_by'] == ())


def test_every_design_level_table_starts_at_an_admitting_level() -> None:
    """
    The year is offered as resolvable on the strength of this.

    An early enough year reaches the oldest design level of whatever
    table governs the site, and the inferer relies on that level
    admitting the unreinforced types in every region.
    """
    tables = dict(CA_DESIGN_LEVELS)
    tables.update(GENERIC_DESIGN_LEVELS)
    legality = get_hazus_unreinforced_design_levels()
    for _breaks, levels in tables.values():
        assert all(levels[0] in allowed for allowed in legality.values())


def _design_level(year: int) -> str:
    """
    Derive the design level a year implies, as the compiler would.

    Args:
        year (int):
            Construction year.

    Returns:
        str:
            The design level for California, UBC zone 4.
    """
    return assign_design_levels(
        pd.Series({'a': year}), state='California', seismic_zone='4'
    )['a']


def _any_value_works(
    build_inventory: Callable[..., AssetInventory],
    inferer: HazusEQStructureTypeInferer,
    features: dict,
    key: str,
    values: list,
) -> bool:
    """
    Say whether any value of one feature makes an asset drawable.

    Args:
        build_inventory (Callable[..., AssetInventory]):
            The inventory factory fixture.
        inferer (HazusEQStructureTypeInferer):
            The inferer to ask.
        features (dict):
            The asset's features.
        key (str):
            The feature to vary. None among the values means removing it.
        values (list):
            The values to try.

    Returns:
        bool:
            True where some value leaves the asset drawable.
    """
    for value in values:
        varied = dict(features)
        if value is None:
            varied.pop(key)
        else:
            varied[key] = value
        if key == 'YearBuilt':
            # The design level is derived from the year, so a change of
            # year carries it along, as it would in the compiler.
            varied['DesignLevel'] = _design_level(value)
        if varied == features:
            continue
        inventory = build_inventory({'a': varied})
        if inferer.check(inventory).loc['a', 'diagnosis'] == DIAGNOSIS_OK:
            return True
    return False


def test_every_resolvable_feature_admits_the_asset_on_its_own(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """
    A feature the caller can change without effect should not be named.

    Sweeps every cell of the region against every general type and, for
    each feature the diagnosis offers, changes that feature alone and
    checks the asset becomes drawable. Then changes each feature the
    diagnosis withholds and checks it does not.
    """
    type_lists, type_weights = get_hazus_occ_type_mapping()
    membership = get_hazus_general_type_membership()
    inferer = _inferer()
    bands = tuple(inferer._bands)
    eras = tuple(type_weights['West Coast']['RES1'])
    occupancies = list(type_weights['West Coast'])
    stories_of = {'low_rise': 2, 'mid_rise': 5, 'high_rise': 9}
    year_of = {'pre_1950': 1930, '1950_to_1970': 1960, 'post_1970': 1990}
    checked = 0
    for occupancy in occupancies:
        for band in bands:
            for era in eras:
                for letter in membership:
                    features = {
                        'OccupancyClass': occupancy,
                        'NumberOfStories': stories_of[band],
                        'YearBuilt': year_of[era],
                        'DesignLevel': _design_level(year_of[era]),
                        'BuildingType': letter,
                    }
                    inventory = build_inventory({'a': features})
                    row = inferer.check(inventory).loc['a']
                    if row['diagnosis'] != DIAGNOSIS_INFEASIBLE:
                        continue
                    checked += 1
                    offered = set(row['resolvable_by'])
                    for key, values in (
                        ('OccupancyClass', occupancies),
                        ('NumberOfStories', list(stories_of.values())),
                        ('YearBuilt', list(year_of.values())),
                        ('BuildingType', [None]),
                    ):
                        works = _any_value_works(
                            build_inventory, inferer, features, key, values
                        )
                        assert works == (key in offered), (
                            occupancy, band, era, letter, key
                        )
    assert checked > 100


def test_a_wood_high_rise_is_diagnosed(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The tables carry no wood above seven stories."""
    inventory = build_inventory({'a': dict(WOOD_TOWER)})
    row = _inferer().check(inventory).loc['a']
    assert row['diagnosis'] == DIAGNOSIS_INFEASIBLE
    assert row['resolvable_by'] == (
        'OccupancyClass', 'NumberOfStories', 'BuildingType'
    )
    assert 'YearBuilt' not in row['resolvable_by']


def test_a_home_without_a_story_count_cannot_change_occupancy(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """
    RES1 is drawn without a story band, and every other class needs one.

    So a change of occupancy resolves nothing for a home that carries no
    story count, while for one that does it opens the other tables.
    """
    inventory = build_inventory(
        {
            'no_stories': dict(HOME, BuildingType='S'),
            'with_stories': dict(HOME, BuildingType='S', NumberOfStories=2),
        }
    )
    verdicts = _inferer().check(inventory)
    assert verdicts.loc['no_stories', 'resolvable_by'] == ('BuildingType',)
    assert verdicts.loc['with_stories', 'resolvable_by'] == (
        'OccupancyClass', 'BuildingType'
    )


def test_a_cell_the_tables_leave_empty_is_diagnosed(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """An eight-story retail building has no row, before any restriction."""
    inventory = build_inventory({'a': dict(STORE, NumberOfStories=8)})
    row = _inferer().check(inventory).loc['a']
    assert row['diagnosis'] == DIAGNOSIS_INFEASIBLE
    assert 'no structure type' in row['detail']
    assert row['resolvable_by'] == ('OccupancyClass', 'NumberOfStories')


@pytest.mark.parametrize(
    'state', ['American Samoa', 'Guam', 'Northern Mariana Islands']
)
def test_a_state_without_a_home_row_is_refused(state: str) -> None:
    """
    A state whose homes cannot be drawn is not one we can serve.

    These three sit in the West Coast group and the Appendix A RES1
    tables carry no row for any of them, so an inventory built for one
    would silently hold no single-family homes.
    """
    with pytest.raises(ValueError, match='no RES1 distribution'):
        _inferer(state=state)


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
    assert expected in _inferer().check(inventory).loc['a', 'detail']


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
    row = _inferer().check(inventory).loc['a']
    assert row['diagnosis'] == DIAGNOSIS_UNSUPPORTED
    assert 'one value per world' in row['detail']


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
    assert verdicts.loc['mid', 'diagnosis'] == DIAGNOSIS_INFEASIBLE


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
    assert verdicts.loc['high', 'diagnosis'] == DIAGNOSIS_INFEASIBLE
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


# ---------------------------------------------------------------------
# unprinted cells


def test_an_unpublished_cell_is_told_apart_from_an_emptied_row() -> None:
    """A cell with no printed row is unpublished, and an emptied one is not."""
    inferer = _inferer()

    assert inferer.cell_is_unpublished(dict(SCHOOL))
    assert inferer.cell_is_unpublished(
        dict(STORE, OccupancyClass='COM7', NumberOfStories=9)
    )
    assert not inferer.cell_is_unpublished(dict(MASONRY_WALKUP))


def test_an_unreadable_driver_has_no_cell_to_ask_about() -> None:
    """A driver that cannot be read selects no cell, and the question raises."""
    features = {
        key: value for key, value in STORE.items() if key != 'DesignLevel'
    }

    with pytest.raises(ValueError, match='no DesignLevel'):
        _inferer().cell_is_unpublished(features)


# ---------------------------------------------------------------------
# completed tables


def test_the_completed_tables_admit_an_unprinted_cell(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The school is infeasible by default and drawable with the completion."""
    inventory = build_inventory({'a': dict(SCHOOL)})
    default = _inferer().check(inventory).loc['a', 'diagnosis']
    completed = _inferer(complete_tables=True).check(inventory)
    assert default == DIAGNOSIS_INFEASIBLE
    assert completed.loc['a', 'diagnosis'] == DIAGNOSIS_OK


def test_a_filled_cell_draws_only_the_types_of_its_row(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """At Pre-Code no type is barred, and the draw stays within the filled row."""
    row = get_completed_west_coast_weights()['EDU1']['mid_rise'][
        '1950_to_1970'
    ]
    types = get_hazus_occ_type_mapping()[0]['West Coast']['mid_rise']
    weighted = {t for t, weight in zip(types, row) if weight > 0}
    features = dict(SCHOOL, DesignLevel='Pre-Code')
    drawn = set(_spread(build_inventory, features, complete_tables=True))
    assert drawn <= weighted
    assert len(drawn) > 1


def test_pinned_draw_in_a_filled_cell(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """One value fixed in place."""
    inventory = build_inventory({'asset_0': dict(SCHOOL)})
    inferer = _inferer(seed=20260828, complete_tables=True)
    assert inferer.draw(inventory)['asset_0'] == 'C2'


def test_a_printed_cell_draws_alike_with_and_without_the_completion(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The completion leaves printed cells as printed."""
    inventory = build_inventory(
        {f'asset_{index}': dict(STORE) for index in range(50)}
    )
    default = _inferer().draw(inventory)
    completed = _inferer(complete_tables=True).draw(inventory)
    assert completed.equals(default)


def test_a_filled_cell_reports_its_fill() -> None:
    """``cell_fill`` returns the route of a filled cell, None for a printed one."""
    inferer = _inferer(complete_tables=True)
    routes = get_west_coast_fill_routes()
    assert inferer.cell_fill(dict(SCHOOL)) == routes[
        ('EDU1', 'mid_rise', '1950_to_1970')
    ]
    assert inferer.cell_fill(dict(STORE)) is None
    assert inferer.cell_fill(dict(HOME)) is None


def test_without_the_completion_no_cell_is_filled() -> None:
    """The inferer then draws every cell from the printed tables."""
    assert _inferer().cell_fill(dict(SCHOOL)) is None


def test_a_filled_cell_is_still_reported_unpublished() -> None:
    """The completion changes what is drawn, not what the tables print."""
    assert _inferer(complete_tables=True).cell_is_unpublished(dict(SCHOOL))


def test_a_filled_row_emptied_by_the_features_is_told_apart(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """
    A five-story wood school is infeasible with the completion on.

    The filled mid-rise row carries no wood. With the completion on,
    withholding the building type resolves it, which it does not with the
    completion off, and ``cell_fill`` reports the cell as filled.
    """
    features = dict(SCHOOL, BuildingType='W')
    inventory = build_inventory({'a': features})
    completed = _inferer(complete_tables=True)
    on = completed.check(inventory).loc['a']
    off = _inferer().check(inventory).loc['a']
    assert on['diagnosis'] == DIAGNOSIS_INFEASIBLE
    assert 'BuildingType' in on['resolvable_by']
    assert 'BuildingType' not in off['resolvable_by']
    assert completed.cell_fill(features) is not None


def test_the_completion_outside_the_west_coast_is_refused() -> None:
    """Only the West Coast tables are completed."""
    with pytest.raises(ValueError, match='West Coast tables'):
        _inferer(state='Iowa', complete_tables=True)
