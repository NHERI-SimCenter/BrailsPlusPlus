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


"""Tests for the replacement cost imputer."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from brails.constants import (
    AREA_FROM_GROSS_AREA,
    AREA_FROM_PLAN_AREA,
    AREA_FROM_STORIES,
    IMPUTATION_DONOR_KEY,
)
from brails.imputers import ObservedDonorImputer, ReplacementCostImputer
from brails.imputers.imputation import Imputation
from brails.imputers.observed_donor_imputer.donor_records import (
    LOG_COLUMNS as DONOR_LOG_COLUMNS,
)
from brails.imputers.observed_donor_imputer.donor_records import (
    OUTCOME_BLOCK_GROUP,
    OUTCOME_DONOR,
    OUTCOME_FEWER_THAN_K,
    OUTCOME_NO_AREA,
    OUTCOME_NO_DONOR,
    ROUND_DONOR,
    ROUND_REPAIR,
)
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    DEFAULT_AREA_WEIGHT,
    DEFAULT_DONOR_CAP_M,
    DEFAULT_DONOR_K,
    DEFAULT_LOCATION_SCALE_M,
    DEFAULT_STORIES_WEIGHT,
    imputation_donors_from_log,
)
from brails.imputers.replacement_cost.replacement_cost_imputer import (
    OUTCOME_IMPUTATION_DONOR,
    OUTCOME_LOCAL_VALUE,
    OUTCOME_NO_CLASS,
    OUTCOME_OWN_RATE,
    OUTCOME_VALUE_CODES,
    ReplacementCostResult,
    footprint_gross_area,
)
from brails.inferers import ACSUnitBudget
from brails.inferers.acs_unit_budget.acs_unit_budget import GEOID_COLUMN
from brails.types.asset_inventory import SOURCE_CODE_NONE, Asset, AssetInventory
from brails.utils.stage_freeze import StageFreeze
from tests.imputers.donor_fixtures import (
    BG_A,
    BG_B,
    OTHER_SEED,
    SEED,
    asset,
    features_of,
    inventory,
    square,
)

SQFT = 0.3048**2

# The columns of an empty B25024 table for ACSUnitBudget.
ACS_COLUMNS = [GEOID_COLUMN] + [
    f'B25024_{line:03d}{kind}' for line in range(2, 11) for kind in 'EM'
]

# The neighbors of labeled_draws on the fixture seed, worlds 0 to 7.
GOLDEN_NEAR = ['n4', 'n1', 'n1', 'n2', 'n2', 'n1', 'n3', 'n3']
GOLDEN_BLOCK_GROUP = ['g1', 'g1', 'g2', 'g1', 'g1', 'g2', 'g2', 'g2']


def source(occupancy: str = 'I', value: str = '-', stories: str = 'I') -> str:
    """
    Return a Hazus earthquake source string.

    Args:
        occupancy (str, optional):
            The occupancy class code. Defaults to ``'I'``.
        value (str, optional):
            The replacement cost code. Defaults to ``'-'``.
        stories (str, optional):
            The story count code. Defaults to ``'I'``.

    Returns:
        str:
            The eight codes: the given occupancy class, stories and value
            codes, and ``-`` for the other five features.
    """
    return f'{occupancy}-{stories}----{value}'


def nsi_building(
    east: float, value: float, gross: float, occupancy: str = 'COM1', **features: Any
) -> tuple[list, dict[str, Any]]:
    """
    Return a one-story building whose occupancy class and value NSI states.

    Args:
        east (float):
            Meters east of the origin.
        value (float):
            Its replacement cost.
        gross (float):
            Its ``GrossArea`` in square feet.
        occupancy (str, optional):
            Its occupancy class. Defaults to ``'COM1'``.
        **features (Any):
            Other features, which override these.

    Returns:
        tuple[list, dict[str, Any]]:
            Coordinates and features.
    """
    return asset(
        east,
        0.0,
        **{
            'OccupancyClass': occupancy,
            'NumberOfStories': 1,
            'ReplacementCost': value,
            'GrossArea': gross,
            'SourceString': source('N', 'N', 'N'),
            **features,
        },
    )


def target(east: float, occupancy: Any = 'COM1', **features: Any) -> tuple:
    """
    Return an unvalued two-story building, occupancy class and stories imputed.

    Args:
        east (float):
            Meters east of the origin.
        occupancy (Any, optional):
            Its occupancy class. Defaults to ``'COM1'``.
        **features (Any):
            Other features, which override these.

    Returns:
        tuple[list, dict[str, Any]]:
            Coordinates and features.
    """
    return asset(
        east,
        0.0,
        **{
            'OccupancyClass': occupancy,
            'NumberOfStories': 2,
            'SourceString': source(),
            **features,
        },
    )


def donor_stage(
    buildings: AssetInventory, donors: dict[str, Any], n_pw: int = 1
) -> AssetInventory:
    """
    Return a donor stage naming the given donor per target.

    Args:
        buildings (AssetInventory):
            The stage, for the coordinates.
        donors (dict[str, Any]):
            Target id to its donor id, or to a list per world.
        n_pw (int, optional):
            The number of worlds. Defaults to 1.

    Returns:
        AssetInventory:
            The donor stage.
    """
    stage = AssetInventory()
    for asset_id, donor in donors.items():
        coordinates = buildings.inventory[asset_id].coordinates
        stage.add_asset(
            asset_id, Asset(asset_id, coordinates, {IMPUTATION_DONOR_KEY: donor})
        )
    stage.n_pw = n_pw
    return stage


def run(
    buildings: AssetInventory,
    donors: AssetInventory,
    world: int | None = None,
    **options: Any,
) -> ReplacementCostResult:
    """
    Freeze a stage and its donor stage and value it on the fixture seed.

    Every log row is checked to hold a value exactly where its outcome has a
    code in ``OUTCOME_VALUE_CODES``.

    Args:
        buildings (AssetInventory):
            The stage.
        donors (AssetInventory):
            The donor stage.
        world (int, optional):
            The world. Defaults to every world.
        **options (Any):
            Overrides of the constructor arguments.

    Returns:
        ReplacementCostResult:
            The result.
    """
    freeze = StageFreeze()
    freeze.freeze('stage', buildings)
    freeze.freeze('donors', donors)
    result = ReplacementCostImputer(**{'seed': SEED, **options}).impute_stage(
        freeze, 'stage', 'donors', world=world
    )
    for row in result.log.itertuples():
        held = result.costs.inventory.get(row.asset_id)
        stored = None if held is None else held.features['ReplacementCost']
        if isinstance(stored, list):
            stored = stored[row.world if world is None else 0]
        assert (stored is not None) == (row.outcome in OUTCOME_VALUE_CODES), row
    return result


def plan_sqft(buildings: AssetInventory, asset_id: str) -> float:
    """
    Return a building's plan area in square feet.

    Args:
        buildings (AssetInventory):
            The inventory.
        asset_id (str):
            The building.

    Returns:
        float:
            The plan area.
    """
    coordinates = buildings.inventory[asset_id].coordinates
    return AssetInventory.centroid_and_plan_area(coordinates)[2] / SQFT


def logged(result: ReplacementCostResult, asset_id: str) -> dict[str, Any]:
    """
    Return the one log row of a building.

    Args:
        result (ReplacementCostResult):
            The result.
        asset_id (str):
            The building.

    Returns:
        dict[str, Any]:
            The row.
    """
    rows = result.log[result.log['asset_id'] == asset_id].to_dict('records')
    assert len(rows) == 1
    return rows[0]


def column(result: ReplacementCostResult, asset_id: str, name: str) -> list[Any]:
    """
    Return one log column of a building, world by world.

    Args:
        result (ReplacementCostResult):
            The result.
        asset_id (str):
            The building.
        name (str):
            The column.

    Returns:
        list[Any]:
            The building's entries in the order of the log.
    """
    return list(result.log.loc[result.log['asset_id'] == asset_id, name])


def value_of(result: ReplacementCostResult, asset_id: str) -> Any:
    """
    Return a building's stored value, or None where it received none.

    Args:
        result (ReplacementCostResult):
            The result.
        asset_id (str):
            The building.

    Returns:
        Any:
            The stored value.
    """
    held = result.costs.inventory.get(asset_id)
    return None if held is None else held.features['ReplacementCost']


class TestRateSources:
    """The rate is the building's own, else its donor's, else a neighbor's."""

    @pytest.mark.parametrize('code', ['N', 'R'])
    def test_own_rate(self, code: str) -> None:
        """A value NSI states gives a rate, times its plan area and story count."""
        buildings = inventory(
            {
                'n': nsi_building(
                    0.0,
                    900.0,
                    3.0,
                    NumberOfStories=4,
                    SourceString=source('N', code),
                )
            }
        )
        result = run(buildings, donor_stage(buildings, {}))
        assert value_of(result, 'n') == pytest.approx(
            300.0 * plan_sqft(buildings, 'n') * 4, rel=1e-12
        )
        row = logged(result, 'n')
        assert row['outcome'] == OUTCOME_OWN_RATE
        assert row['rate'] == 300.0
        assert pd.isna(row['donor_id'])

    def test_own_rate_before_the_donor(self) -> None:
        """A building with its own rate ignores the donor the stage names."""
        buildings = inventory(
            {'d': nsi_building(10.0, 500.0, 1.0), 'n': nsi_building(0.0, 100.0, 1.0)}
        )
        result = run(buildings, donor_stage(buildings, {'n': 'd'}))
        assert logged(result, 'n')['rate'] == 100.0

    def test_donor_rate(self) -> None:
        """A target takes its donor's rate times its plan area and story count."""
        buildings = inventory(
            {'d': nsi_building(500.0, 450000.0, 1800.0), 't': target(0.0)}
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}))
        expected = 450000.0 / 1800.0 * plan_sqft(buildings, 't') * 2
        assert value_of(result, 't') == pytest.approx(expected, rel=1e-12)
        row = logged(result, 't')
        assert row['outcome'] == OUTCOME_IMPUTATION_DONOR
        assert (row['donor_id'], row['rate']) == ('d', 250.0)
        assert math.isnan(row['distance_m'])

    def test_donor_rate_before_the_neighbor(self) -> None:
        """A named donor farther away and of another occupancy class still wins."""
        buildings = inventory(
            {
                'near': nsi_building(10.0, 100.0, 1.0),
                'far': nsi_building(1500.0, 300.0, 1.0, occupancy='RES3A'),
                't': target(0.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {'t': 'far'}))
        assert logged(result, 't')['rate'] == 300.0

    def test_redrawn_donor_is_used(self) -> None:
        """Where validation re-drew, the re-draw's donor gives the rate."""
        buildings = inventory(
            {
                'first': nsi_building(10.0, 100.0, 1.0),
                'second': nsi_building(20.0, 200.0, 1.0),
                't': target(0.0),
            }
        )
        rows = [
            {
                **dict.fromkeys(DONOR_LOG_COLUMNS),
                'asset_id': 't',
                'world': 0,
                'donor_id': donor,
                'values': {'OccupancyClass': 'COM1'},
                'outcome': OUTCOME_DONOR,
                'round': round_number,
            }
            for donor, round_number in (
                ('first', ROUND_DONOR),
                ('second', ROUND_REPAIR),
            )
        ]
        log = pd.DataFrame(rows, columns=list(DONOR_LOG_COLUMNS))
        targets = inventory({'t': target(0.0)})
        result = run(buildings, imputation_donors_from_log(log, targets, n_worlds=1))
        assert logged(result, 't')['donor_id'] == 'second'
        assert logged(result, 't')['rate'] == 200.0

    def test_neighbor_rate(self) -> None:
        """A target without a donor takes a same-class neighbor's rate."""
        buildings = inventory(
            {'c': nsi_building(900.0, 40.0, 1.0), 't': target(0.0)}
        )
        result = run(buildings, donor_stage(buildings, {}))
        row = logged(result, 't')
        assert (row['outcome'], row['donor_id']) == (OUTCOME_FEWER_THAN_K, 'c')
        assert row['distance_m'] == pytest.approx(900.0, rel=1e-3)
        assert value_of(result, 't') == pytest.approx(
            40.0 * plan_sqft(buildings, 't') * 2
        )

    @pytest.mark.parametrize(
        'donor',
        [
            {'GrossArea': None},
            {'GrossArea': 0.0},
            {'ReplacementCost': math.inf},
            {'SourceString': source('N', 'I')},
            {'SourceString': source('N', '-'), 'ReplacementCost': 'NA'},
        ],
    )
    def test_named_donor_without_a_rate(self, donor: dict[str, Any]) -> None:
        """A donor without a positive NSI value and GrossArea means a draw."""
        buildings = inventory(
            {
                'd': nsi_building(10.0, 100.0, 1.0, **donor),
                'c': nsi_building(900.0, 40.0, 1.0),
                't': target(0.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}))
        assert logged(result, 't')['donor_id'] == 'c'
        assert logged(result, 't')['rate'] == 40.0

    def test_own_rate_whatever_the_class_code(self) -> None:
        """A building whose occupancy class is coded I or D has its own rate."""
        buildings = inventory(
            {
                'i': nsi_building(0.0, 100.0, 1.0, SourceString=source('I', 'N')),
                'd': nsi_building(30.0, 100.0, 1.0, SourceString=source('D', 'R')),
            }
        )
        result = run(buildings, donor_stage(buildings, {}))
        assert set(result.log['outcome']) == {OUTCOME_OWN_RATE}

    @pytest.mark.parametrize('code', ['I', 'D', '-'])
    def test_value_of_another_source_gives_no_rate(self, code: str) -> None:
        """A value coded I, D or - is replaced, and never read as a rate."""
        buildings = inventory(
            {
                'c': nsi_building(900.0, 40.0, 1.0),
                'x': nsi_building(0.0, 999.0, 1.0, SourceString=source('N', code)),
            }
        )
        result = run(buildings, donor_stage(buildings, {}))
        assert logged(result, 'x')['donor_id'] == 'c'
        assert value_of(result, 'x') == pytest.approx(
            40.0 * plan_sqft(buildings, 'x')
        )

    def test_local_value_kept(self) -> None:
        """A value coded L is kept unchanged, and its row logs its own rate."""
        buildings = inventory(
            {'l': nsi_building(0.0, 999, 3.0, SourceString=source('N', 'L'))}
        )
        result = run(buildings, donor_stage(buildings, {}))
        assert type(value_of(result, 'l')) is int
        assert value_of(result, 'l') == 999
        row = logged(result, 'l')
        assert (row['outcome'], row['rate']) == (OUTCOME_LOCAL_VALUE, 333.0)

    def test_local_donor_rate(self) -> None:
        """A local value over its GrossArea is a donor's rate in every world."""
        buildings = inventory(
            {
                'c': nsi_building(900.0, 40.0, 1.0),
                'l': nsi_building(
                    0.0,
                    999.0,
                    3.0,
                    NumberOfStories=[1, 3],
                    SourceString=source('N', 'L'),
                ),
                't': target(5.0),
            }
        )
        buildings.n_pw = 2
        result = run(buildings, donor_stage(buildings, {'t': 'l'}))
        assert value_of(result, 'l') == 999.0
        assert column(result, 'l', 'outcome') == [OUTCOME_LOCAL_VALUE] * 2
        assert column(result, 't', 'outcome') == [OUTCOME_IMPUTATION_DONOR] * 2
        assert column(result, 't', 'donor_id') == ['l', 'l']
        assert column(result, 't', 'rate') == [333.0, 333.0]
        assert value_of(result, 't') == pytest.approx(
            333.0 * plan_sqft(buildings, 't') * 2, rel=1e-12
        )

    def test_local_value_without_gross_area(self) -> None:
        """A local value without GrossArea gives no rate, and its target draws."""
        buildings = inventory(
            {
                'c': nsi_building(900.0, 40.0, 1.0),
                'l': nsi_building(0.0, 999.0, None, SourceString=source('N', 'L')),
                't': target(5.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {'t': 'l'}))
        assert value_of(result, 'l') == 999.0
        row = logged(result, 'l')
        assert row['outcome'] == OUTCOME_LOCAL_VALUE
        assert math.isnan(row['rate'])
        row = logged(result, 't')
        assert (row['outcome'], row['donor_id'], row['rate']) == (
            OUTCOME_FEWER_THAN_K,
            'c',
            40.0,
        )

    def test_no_named_donor(self) -> None:
        """A target the donor stage lacks, or names no donor for, draws."""
        buildings = inventory(
            {
                'c': nsi_building(900.0, 40.0, 1.0),
                't': target(0.0),
                'u': target(30.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {'u': [None]}))
        assert {logged(result, a)['donor_id'] for a in ('t', 'u')} == {'c'}

    def test_rates_come_from_the_stage(self) -> None:
        """A donor's rate is its stated value's, not the value the step wrote."""
        buildings = inventory(
            {
                'unreached': target(10.0),
                'withheld': nsi_building(
                    20.0, 'NA', 1.0, SourceString=source('I', '-')
                ),
                'd': nsi_building(30.0, 100.0, 1.0, 'RES3A', NumberOfStories=5),
                'c': nsi_building(900.0, 40.0, 1.0),
                't': target(0.0),
                'u': target(5.0),
                'v': target(40.0),
            }
        )
        result = run(
            buildings,
            donor_stage(buildings, {'t': 'unreached', 'u': 'withheld', 'v': 'd'}),
        )
        assert {logged(result, a)['donor_id'] for a in ('t', 'u')} == {'c'}
        assert value_of(result, 'unreached') is not None
        assert logged(result, 'v')['rate'] == 100.0

    def test_donor_chain_not_followed(self) -> None:
        """A target whose donor took a donor's rate draws a neighbor instead."""
        buildings = inventory(
            {
                'd': nsi_building(30.0, 100.0, 1.0, 'RES3A'),
                'c': nsi_building(900.0, 40.0, 1.0),
                'a': target(0.0, SourceString=source('L')),
                'b': target(10.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {'a': 'd', 'b': 'a'}))
        assert logged(result, 'a')['outcome'] == OUTCOME_IMPUTATION_DONOR
        row = logged(result, 'b')
        assert (row['outcome'], row['donor_id'], row['rate']) == (
            OUTCOME_FEWER_THAN_K,
            'c',
            40.0,
        )


class TestFootprintGrossArea:
    """The gross area from the footprint is the plan area times the story count."""

    @pytest.mark.parametrize('gross', [7.0, 1.0e6, None])
    def test_gross_area_feature_never_multiplies_a_rate(self, gross: Any) -> None:
        """GrossArea divides a building's own value and never multiplies a rate."""
        buildings = inventory(
            {
                'n': nsi_building(10.0, 100.0, 1.0, NumberOfStories=3),
                't': target(0.0, GrossArea=gross, NumberOfStories=4),
            }
        )
        result = run(buildings, donor_stage(buildings, {'t': 'n'}))
        for asset_id, stories in (('n', 3), ('t', 4)):
            row = logged(result, asset_id)
            assert row['area_source'] == AREA_FROM_STORIES
            assert row['footprint_gross_area_sqft'] == pytest.approx(
                plan_sqft(buildings, asset_id) * stories
            )
            assert value_of(result, asset_id) == pytest.approx(
                100.0 * row['footprint_gross_area_sqft']
            )

    def test_value_follows_the_worlds_stories(self) -> None:
        """Each world's value scales with that world's story count."""
        buildings = inventory(
            {
                'n': nsi_building(10.0, 100.0, 1.0, NumberOfStories=[1, 2, 3]),
                't': target(0.0, NumberOfStories=[3, 1, 2]),
            }
        )
        buildings.n_pw = 3
        result = run(buildings, donor_stage(buildings, {'t': 'n'}))
        for asset_id, stories in (('n', [1, 2, 3]), ('t', [3, 1, 2])):
            plan = plan_sqft(buildings, asset_id)
            assert value_of(result, asset_id) == pytest.approx(
                [100.0 * plan * s for s in stories]
            )

    @pytest.mark.parametrize('stories', [None, 0, -1, math.nan, True, 'two'])
    def test_plan_area_as_one_story(self, stories: Any) -> None:
        """Without a usable story count the plan area counts as one story."""
        buildings = inventory(
            {
                'd': nsi_building(10.0, 100.0, 1.0),
                't': target(0.0, NumberOfStories=stories, GrossArea=50.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}))
        row = logged(result, 't')
        assert row['area_source'] == AREA_FROM_PLAN_AREA
        assert row['footprint_gross_area_sqft'] == pytest.approx(
            plan_sqft(buildings, 't')
        )
        assert value_of(result, 't') == pytest.approx(
            100.0 * row['footprint_gross_area_sqft']
        )

    def test_no_area(self) -> None:
        """A building without a plan area gets no value, its GrossArea unread."""
        lon, lat = square(0.0, 0.0)[0]
        buildings = AssetInventory()
        buildings.add_asset(
            'n',
            Asset(
                'n',
                [[lon, lat]],
                {
                    'OccupancyClass': 'COM1',
                    'ReplacementCost': 100.0,
                    'GrossArea': 1.0,
                    'SourceString': source('N', 'N'),
                },
            ),
        )
        result = run(buildings, donor_stage(buildings, {}))
        row = logged(result, 'n')
        assert row['outcome'] == OUTCOME_NO_AREA
        assert value_of(result, 'n') is None


class TestUnitBudgetGrossArea:
    """The gross area is the unit budget's on a footprint without GrossArea."""

    def test_same_area_and_source(self) -> None:
        """Gross area and its source match the unit budget's on these footprints."""
        # not covered: a footprint with neither area nor gross area, which the
        # budget gives a zero area, and unreadable coordinates, on which it raises
        records = {
            'stories': target(30.0, NumberOfStories=3),
            'numpy': target(60.0, NumberOfStories=np.int64(2)),
            'float': target(90.0, NumberOfStories=2.5),
            'nothing': target(120.0, NumberOfStories=None),
            'zero_gross': target(150.0, GrossArea=0.0, NumberOfStories=2),
            'nan_gross': target(180.0, GrossArea=math.nan, NumberOfStories=2),
            'bool': target(210.0, NumberOfStories=True),
            'text': target(240.0, NumberOfStories='NA'),
        }
        buildings = inventory(records)
        budget = ACSUnitBudget(
            pd.DataFrame(columns=ACS_COLUMNS),
            [],
            imputer=ObservedDonorImputer(seed=1),
        )
        sites = budget._sites(buildings)
        for asset_id in records:
            area, source_name = footprint_gross_area(
                asset_id, buildings.inventory[asset_id]
            )
            site = sites[asset_id]
            assert source_name == site.gross_source, asset_id
            assert area == pytest.approx(site.gross_m2 / SQFT, rel=1e-12), asset_id

    def test_departure_on_gross_area(self) -> None:
        """The budget reads GrossArea and the imputer does not."""
        buildings = inventory({'g': target(0.0, GrossArea=500.0, NumberOfStories=3)})
        budget = ACSUnitBudget(
            pd.DataFrame(columns=ACS_COLUMNS),
            [],
            imputer=ObservedDonorImputer(seed=1),
        )
        assert budget._sites(buildings)['g'].gross_source == AREA_FROM_GROSS_AREA
        area, source_name = footprint_gross_area('g', buildings.inventory['g'])
        assert source_name == AREA_FROM_STORIES
        assert area == pytest.approx(plan_sqft(buildings, 'g') * 3)


class TestOutcomes:
    """Buildings that get no value are logged with the reason."""

    @pytest.mark.parametrize('occupancy', [None, 'NA', '', 3])
    def test_no_class(self, occupancy: Any) -> None:
        """Without an occupancy class, own rate or donor's rate, it gets none."""
        buildings = inventory(
            {
                'c': nsi_building(10.0, 100.0, 1.0),
                't': target(0.0, occupancy=occupancy),
            }
        )
        result = run(buildings, donor_stage(buildings, {}))
        row = logged(result, 't')
        assert row['outcome'] == OUTCOME_NO_CLASS
        assert value_of(result, 't') is None

    def test_class_read_only_by_the_neighbor_draw(self) -> None:
        """The occupancy class is read only by the neighbor draw."""
        buildings = inventory(
            {
                'n': nsi_building(10.0, 100.0, 1.0, OccupancyClass=None),
                't': target(0.0, occupancy=None),
            }
        )
        result = run(buildings, donor_stage(buildings, {'t': 'n'}))
        assert logged(result, 'n')['outcome'] == OUTCOME_OWN_RATE
        assert logged(result, 't')['outcome'] == OUTCOME_IMPUTATION_DONOR

    def test_no_rate(self) -> None:
        """Without a rate a building gets no value, and the log says so."""
        buildings = inventory(
            {
                'c': nsi_building(10.0, 40.0, 1.0),
                'ind': target(30.0, occupancy='IND1'),
                'x': nsi_building(60.0, 99.0, 1.0, occupancy='IND1', GrossArea=0.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {}))
        for asset_id in ('ind', 'x'):
            row = logged(result, asset_id)
            assert row['outcome'] == OUTCOME_NO_DONOR
            assert pd.isna(row['donor_id'])
            assert math.isnan(row['rate'])
        assert result.costs.get_asset_ids() == ['c']


class TestReport:
    """Each log row's outcome gives its value's code, per building and world."""

    def test_value_code_per_building_and_world(self) -> None:
        """The outcome gives each world's value code."""
        buildings = inventory(
            {
                'n': nsi_building(10.0, 100.0, 1.0),
                'c': nsi_building(900.0, 40.0, 1.0, occupancy='RES1'),
                't': target(0.0, occupancy=['COM1', 'RES1', 'IND1']),
            }
        )
        buildings.n_pw = 3
        result = run(
            buildings, donor_stage(buildings, {'t': ['n', None, None]}, n_pw=3)
        )
        codes = {
            a: [
                OUTCOME_VALUE_CODES.get(o, SOURCE_CODE_NONE)
                for o in column(result, a, 'outcome')
            ]
            for a in ('n', 't')
        }
        assert codes == {'n': ['R'] * 3, 't': ['I', 'I', SOURCE_CODE_NONE]}
        assert column(result, 't', 'outcome')[1] == OUTCOME_FEWER_THAN_K
        assert column(result, 't', 'donor_id')[:2] == ['n', 'c']
        assert column(result, 't', 'world') == [0, 1, 2]
        plan = plan_sqft(buildings, 't')
        stored = value_of(result, 't')
        assert stored[:2] == pytest.approx([200.0 * plan, 80.0 * plan])
        assert stored[2] is None

    def test_one_row_per_building_and_world(self) -> None:
        """The log holds every building of the stage in every world."""
        buildings = inventory(
            {'d': nsi_building(10.0, 100.0, 1.0), 't': target(0.0)}
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}, n_pw=4))
        assert len(result.log) == 8
        assert set(zip(result.log['asset_id'], result.log['world'])) == {
            (a, w) for a in ('d', 't') for w in range(4)
        }


class TestSameClassNeighborDraw:
    """A building without an own or donor's rate draws one of its occupancy class."""

    def test_class_filter(self) -> None:
        """A nearer building of another occupancy class is never drawn."""
        buildings = inventory(
            {
                'res': nsi_building(10.0, 999.0, 1.0, occupancy='RES1'),
                'c': nsi_building(1500.0, 40.0, 1.0),
                't': target(0.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {}, n_pw=20))
        assert set(column(result, 't', 'donor_id')) == {'c'}

    def test_class_of_each_world(self) -> None:
        """A target draws from the occupancy class it has in each world."""
        buildings = inventory(
            {
                'com': nsi_building(10.0, 40.0, 1.0),
                'res': nsi_building(20.0, 60.0, 1.0, occupancy='RES1'),
                't': target(0.0, occupancy=['COM1', 'RES1']),
            }
        )
        buildings.n_pw = 2
        result = run(buildings, donor_stage(buildings, {}))
        assert column(result, 't', 'donor_id') == ['com', 'res']

    @pytest.mark.parametrize(
        'excluded',
        [
            {'SourceString': source('I', 'N')},
            {'SourceString': source('D', 'N')},
            {'GrossArea': None},
            {'GrossArea': 0.0},
            {'SourceString': source('N', 'I')},
            {'SourceString': source('N', 'L'), 'GrossArea': None},
        ],
    )
    def test_buildings_never_drawn(self, excluded: dict[str, Any]) -> None:
        """Occupancy class coded I or D, value coded I or no GrossArea bars it."""
        buildings = inventory(
            {
                'x': nsi_building(10.0, 999.0, 1.0, **excluded),
                'c': nsi_building(1500.0, 40.0, 1.0),
                't': target(0.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {}, n_pw=20))
        assert set(column(result, 't', 'donor_id')) == {'c'}

    def test_local_and_footprint_classes_are_drawn(self) -> None:
        """A local or footprint occupancy class makes a building a candidate."""
        buildings = inventory(
            {
                'l': nsi_building(10.0, 40.0, 1.0, SourceString=source('L', 'N')),
                'f': nsi_building(20.0, 40.0, 1.0, SourceString=source('F', 'R')),
                't': target(0.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {}, n_pw=30))
        assert set(column(result, 't', 'donor_id')) == {'l', 'f'}

    def test_local_value_is_drawn(self) -> None:
        """A local value over its GrossArea makes its building a candidate."""
        buildings = inventory(
            {
                'l': nsi_building(
                    10.0,
                    999.0,
                    3.0,
                    NumberOfStories=3,
                    SourceString=source('N', 'L'),
                ),
                't': target(0.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {}))
        row = logged(result, 't')
        assert (row['outcome'], row['donor_id'], row['rate']) == (
            OUTCOME_FEWER_THAN_K,
            'l',
            333.0,
        )
        assert value_of(result, 't') == pytest.approx(
            333.0 * plan_sqft(buildings, 't') * 2, rel=1e-12
        )

    def test_block_group_fallback(self) -> None:
        """Without a building within the cap, one of the block group is drawn."""
        buildings = inventory(
            {
                'far': nsi_building(3000.0, 40.0, 1.0, BLOCKGROUP_GEOID=BG_A),
                'other': nsi_building(-2500.0, 99.0, 1.0, BLOCKGROUP_GEOID=BG_B),
                't': target(0.0, BLOCKGROUP_GEOID=BG_A),
            }
        )
        result = run(buildings, donor_stage(buildings, {}))
        row = logged(result, 't')
        assert (row['outcome'], row['donor_id']) == (OUTCOME_BLOCK_GROUP, 'far')

    @pytest.mark.parametrize(
        ('area_weight', 'expected'), [(0.0, 'big'), (1.0, 'same')]
    )
    def test_area_weight_changes_the_neighbor(
        self, area_weight: float, expected: str
    ) -> None:
        """An area weight prefers a neighbor of the target's plan area."""
        big_features = nsi_building(3.0, 10.0, 1.0)[1]
        buildings = inventory(
            {
                'big': (square(3.0, 0.0, side=100.0), big_features),
                'same': nsi_building(5.0, 20.0, 1.0),
                't': target(0.0),
            }
        )
        result = run(
            buildings,
            donor_stage(buildings, {}, n_pw=20),
            k=1,
            area_weight=area_weight,
        )
        assert set(column(result, 't', 'donor_id')) == {expected}

    @pytest.mark.parametrize(('scale', 'expected'), [(100.0, 'same'), (50.0, 'big')])
    def test_location_scale_changes_the_neighbor(
        self, scale: float, expected: str
    ) -> None:
        """A shorter location scale makes distance outweigh the area difference."""
        big_features = nsi_building(3.0, 10.0, 1.0)[1]
        buildings = inventory(
            {
                'big': (square(3.0, 0.0, side=100.0), big_features),
                'same': nsi_building(400.0, 20.0, 1.0),
                't': target(0.0),
            }
        )
        result = run(
            buildings,
            donor_stage(buildings, {}, n_pw=20),
            k=1,
            location_scale_m=scale,
        )
        assert set(column(result, 't', 'donor_id')) == {expected}

    @pytest.mark.parametrize(
        ('stories_weight', 'expected'), [(0.0, 'one'), (1.0, 'two')]
    )
    def test_stories_weight_changes_the_neighbor(
        self, stories_weight: float, expected: str
    ) -> None:
        """A story weight prefers a neighbor of the target's story count."""
        buildings = inventory(
            {
                'one': nsi_building(5.0, 10.0, 1.0),
                'two': nsi_building(50.0, 20.0, 1.0, NumberOfStories=2),
                't': target(0.0),
            }
        )
        result = run(
            buildings,
            donor_stage(buildings, {}, n_pw=20),
            k=1,
            stories_weight=stories_weight,
        )
        assert set(column(result, 't', 'donor_id')) == {expected}

    @pytest.mark.parametrize('side', ['target', 'candidate'])
    def test_story_term_without_a_count(self, side: str) -> None:
        """A side without a story count adds no story term."""
        near = nsi_building(5.0, 10.0, 1.0, NumberOfStories=3)
        far = nsi_building(50.0, 20.0, 1.0)
        t = target(0.0, NumberOfStories='NA')
        if side == 'candidate':
            near[1].pop('NumberOfStories')
            far[1]['NumberOfStories'] = 3
            t = target(0.0, NumberOfStories=3)
        buildings = inventory({'near': near, 'far': far, 't': t})
        result = run(buildings, donor_stage(buildings, {}), k=1)
        assert logged(result, 't')['donor_id'] == 'near'

    def test_k_and_defaults(self) -> None:
        """With k 1 the nearest is drawn, and defaults are the donor constants."""
        buildings = inventory(
            {
                **{
                    f'c{i}': nsi_building(100.0 * i, 10.0 * i, 1.0)
                    for i in range(1, 4)
                },
                't': target(0.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {}, n_pw=30), k=1)
        assert set(column(result, 't', 'donor_id')) == {'c1'}
        default = ReplacementCostImputer()
        assert (
            default.k,
            default.cap_m,
            default.location_scale_m,
            default.area_weight,
            default.stories_weight,
        ) == (
            DEFAULT_DONOR_K,
            DEFAULT_DONOR_CAP_M,
            DEFAULT_LOCATION_SCALE_M,
            DEFAULT_AREA_WEIGHT,
            DEFAULT_STORIES_WEIGHT,
        )

    def test_seed_recorded(self) -> None:
        """Without a seed one is drawn, recorded and returned."""
        freeze = StageFreeze()
        freeze.freeze('stage', inventory({'t': target(0.0)}))
        freeze.freeze('donors', AssetInventory())
        imputer = ReplacementCostImputer()
        result = imputer.impute_stage(freeze, 'stage', 'donors')
        assert isinstance(imputer.seed, int)
        assert result.seed == imputer.seed


def labeled_draws(
    seed: int = SEED,
    order: list[str] | None = None,
    extra: dict[str, tuple[list, dict[str, Any]]] | None = None,
    n_worlds: int = 8,
) -> dict[str, list[Any]]:
    """
    Return the neighbors of two targets, one per label, in every world.

    ``near`` has five candidates within the cap and draws under the donor
    label. ``bg`` lies in block group A, beyond the cap of every candidate,
    and draws under the fallback label from three buildings of that block
    group.

    Args:
        seed (int, optional):
            The seed. Defaults to the fixture seed.
        order (list[str], optional):
            The insertion order of the stage. Defaults to the fixture's.
        extra (dict, optional):
            Buildings added to the stage.
        n_worlds (int, optional):
            The number of worlds. Defaults to 8.

    Returns:
        dict[str, list[Any]]:
            Target id to its neighbor per world.
    """
    records = {
        **{f'n{i}': nsi_building(100.0 * i, 10.0 * i, 1.0) for i in range(1, 6)},
        **{
            f'g{i}': nsi_building(
                -2500.0 - 100.0 * i, 10.0 * i, 1.0, BLOCKGROUP_GEOID=BG_A
            )
            for i in range(1, 4)
        },
        'near': target(0.0),
        'bg': target(-20000.0, BLOCKGROUP_GEOID=BG_A),
        **(extra or {}),
    }
    buildings = inventory(records, order)
    result = run(buildings, donor_stage(buildings, {}, n_pw=n_worlds), seed=seed)
    return {a: column(result, a, 'donor_id') for a in ('near', 'bg')}


class TestLabels:
    """The two draws are keyed by identity under labels of their own."""

    def test_golden(self) -> None:
        """Each label gives the pinned neighbors on the fixture seed."""
        assert labeled_draws() == {
            'near': GOLDEN_NEAR,
            'bg': GOLDEN_BLOCK_GROUP,
        }

    def test_seed_sensitivity(self) -> None:
        """Another seed changes at least one pick under each label."""
        base, other = labeled_draws(), labeled_draws(seed=OTHER_SEED)
        assert base['near'] != other['near']
        assert base['bg'] != other['bg']

    def test_reordering(self) -> None:
        """The stage's insertion order changes nothing."""
        base = labeled_draws()
        ids = [*(f'n{i}' for i in range(1, 6)), *(f'g{i}' for i in range(1, 4))]
        order = ['bg', 'near', *reversed(ids)]
        assert labeled_draws(order=order) == base

    def test_locality_within_the_cap(self) -> None:
        """Buildings beyond the cap leave a draw with a candidate within it alone."""
        base = labeled_draws()
        extra = {
            'beyond': nsi_building(2600.0, 5.0, 1.0),
            'res_beyond': nsi_building(-2600.0, 5.0, 1.0, occupancy='RES1'),
        }
        assert labeled_draws(extra=extra)['near'] == base['near']

    def test_locality_of_the_fallback(self) -> None:
        """Buildings beyond the cap and outside the block group leave it alone."""
        base = labeled_draws()
        extra = {'outside': nsi_building(-17000.0, 5.0, 1.0, BLOCKGROUP_GEOID=BG_B)}
        assert labeled_draws(extra=extra)['bg'] == base['bg']

    @pytest.mark.parametrize(
        ('block_group', 'expected'), [(BG_B, None), (BG_A, 'edge')]
    )
    def test_just_beyond_the_cap(self, block_group: str, expected: Any) -> None:
        """A building just past the cap is drawn only in the target's block group."""
        buildings = inventory(
            {
                'edge': nsi_building(
                    2001.0, 40.0, 1.0, BLOCKGROUP_GEOID=block_group
                ),
                't': target(0.0, BLOCKGROUP_GEOID=BG_A),
            }
        )
        result = run(buildings, donor_stage(buildings, {}, n_pw=10))
        assert set(column(result, 't', 'donor_id')) == {expected}


class TestImpute:
    """The library entry point values every building without a donor stage."""

    def test_plain_inventory(self) -> None:
        """Values and codes are rewritten; without a rate both go."""
        plain = inventory(
            {
                'c': nsi_building(500.0, 40.0, 1.0),
                't': target(0.0),
                'x': nsi_building(
                    10.0, 77.0, 1.0, occupancy='IND1', SourceString=source('N', 'I')
                ),
                'bare': asset(30.0, 0.0, OccupancyClass='COM1', NumberOfStories=1),
                'loc': nsi_building(60.0, 123.0, 1.0, SourceString=source('N', 'L')),
            }
        )
        before = features_of(plain)
        imputer = ReplacementCostImputer(seed=SEED)
        output = imputer.impute(plain)
        assert isinstance(imputer, Imputation)
        assert features_of(plain) == before
        assert output.n_pw == 1
        staged = run(plain, AssetInventory())
        for asset_id, code in (('c', 'R'), ('t', 'I')):
            features = output.inventory[asset_id].features
            assert features['ReplacementCost'] == value_of(staged, asset_id)
            assert features['SourceString'][7] == code
        assert 'ReplacementCost' not in output.inventory['x'].features
        assert output.inventory['x'].features['SourceString'] == source('N', '-')
        bare = output.inventory['bare'].features
        assert bare['ReplacementCost'] == value_of(staged, 'bare')
        assert 'SourceString' not in bare
        loc = output.inventory['loc'].features
        assert (loc['ReplacementCost'], loc['SourceString']) == (
            123.0,
            source('N', 'L'),
        )

    def test_codes_per_world(self) -> None:
        """A building valued in one world and not another gets a code per world."""
        plain = inventory(
            {
                'c': nsi_building(10.0, 40.0, 1.0),
                't': target(0.0, occupancy=['COM1', 'IND1']),
            }
        )
        plain.n_pw = 2
        output = ReplacementCostImputer(seed=SEED).impute(plain)
        features = output.inventory['t'].features
        assert [s[7] for s in features['SourceString']] == ['I', '-']
        assert features['ReplacementCost'][1] is None
        assert output.inventory['c'].features['SourceString'] == source(
            'N', 'R', 'N'
        )


class TestWorlds:
    """Each world reads its own realization of both stages."""

    def three_worlds(self) -> tuple[AssetInventory, AssetInventory]:
        """
        Return a three-world stage and donor stage.

        Returns:
            tuple[AssetInventory, AssetInventory]:
                The stage, whose target's stories and occupancy class
                differ by world, and the donor stage, whose donor does.
        """
        buildings = inventory(
            {
                'a': nsi_building(10.0, 100.0, 1.0),
                'b': nsi_building(20.0, 300.0, 1.0),
                't': target(
                    0.0, NumberOfStories=[1, 2, 3], occupancy=['COM1', None, 'COM4']
                ),
            }
        )
        buildings.n_pw = 3
        return buildings, donor_stage(buildings, {'t': ['a', None, 'b']}, n_pw=3)

    def test_per_world_values(self) -> None:
        """Every world is computed, and a world without a value holds None."""
        buildings, donors = self.three_worlds()
        result = run(buildings, donors)
        plan = plan_sqft(buildings, 't')
        stored = value_of(result, 't')
        assert stored[0] == pytest.approx(100.0 * plan)
        assert stored[1] is None
        assert stored[2] == pytest.approx(900.0 * plan)
        assert result.costs.n_pw == 3
        assert column(result, 't', 'outcome') == [
            OUTCOME_IMPUTATION_DONOR,
            OUTCOME_NO_CLASS,
            OUTCOME_IMPUTATION_DONOR,
        ]

    def test_one_world_equals_its_realization(self) -> None:
        """World 2 of the stage equals a run on scalar stages of that realization."""
        buildings, donors = self.three_worlds()
        whole = run(buildings, donors)
        one = run(buildings, donors, world=2)
        scalar = run(
            buildings.get_world_realization(2),
            donors.get_world_realization(2),
            world=2,
        )
        assert one.costs.n_pw == 1
        for result in (one, scalar):
            assert value_of(result, 't') == value_of(whole, 't')[2]
            pd.testing.assert_frame_equal(
                result.log,
                whole.log[whole.log['world'] == 2].reset_index(drop=True),
            )

    def test_neighbor_draw_of_one_world(self) -> None:
        """World 2's neighbor draw equals a run on that realization with world 2."""
        buildings = inventory(
            {
                **{
                    f'c{i}': nsi_building(100.0 * i, 10.0 * i, 1.0)
                    for i in range(1, 6)
                },
                't': target(0.0),
            }
        )
        buildings.n_pw = 3
        donors = donor_stage(buildings, {'t': None}, n_pw=3)
        whole = run(buildings, donors)
        scalar = run(
            buildings.get_world_realization(2),
            donors.get_world_realization(2),
            world=2,
        )
        assert value_of(scalar, 't') == value_of(whole, 't')[2]
        assert logged(scalar, 't')['donor_id'] == column(whole, 't', 'donor_id')[2]
        assert len(set(column(whole, 't', 'donor_id'))) > 1

    def test_agreeing_worlds_collapse(self) -> None:
        """A value equal in every world is stored as a scalar."""
        buildings = inventory(
            {'d': nsi_building(10.0, 100.0, 1.0), 't': target(0.0)}
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}, n_pw=4))
        assert value_of(result, 't') == pytest.approx(
            100.0 * plan_sqft(buildings, 't') * 2
        )
        assert result.costs.n_pw == 4


class TestStagesAndArguments:
    """What impute_stage reads and refuses."""

    def test_stages_unchanged(self) -> None:
        """The manifest passes after a run, with the digests of the freeze."""
        buildings = inventory(
            {
                'd': nsi_building(10.0, 100.0, 1.0),
                't': target(0.0),
                'u': target(30.0),
            }
        )
        freeze = StageFreeze()
        digests = [
            freeze.freeze('stage', buildings),
            freeze.freeze('donors', donor_stage(buildings, {'t': 'd'})),
        ]
        ReplacementCostImputer().impute_stage(freeze, 'stage', 'donors')
        assert [s['digest'] for s in freeze.manifest()['stages']] == digests

    def test_donor_stage_required(self) -> None:
        """No donor stage raises TypeError, and an unfrozen one LookupError."""
        freeze = StageFreeze()
        freeze.freeze('stage', inventory({'t': target(0.0)}))
        with pytest.raises(TypeError, match='donor stage'):
            ReplacementCostImputer().impute_stage(freeze, 'stage', None)
        with pytest.raises(LookupError):
            ReplacementCostImputer().impute_stage(freeze, 'stage', 'donors')

    @pytest.mark.parametrize(
        'change', [{'SourceString': None}, {'SourceString': 'NNN'}]
    )
    def test_source_string_of_a_value(self, change: dict[str, Any]) -> None:
        """A building holding a value without a full source string raises."""
        buildings = inventory(
            {'d': nsi_building(10.0, 100.0, 1.0, **change), 't': target(0.0)}
        )
        with pytest.raises(ValueError, match='no source string'):
            run(buildings, donor_stage(buildings, {}))

    @pytest.mark.parametrize('stray', ['gone', 0])
    def test_donor_stage_of_other_assets(self, stray: Any) -> None:
        """A donor stage holding an asset the stage lacks raises."""
        buildings = inventory(
            {'d': nsi_building(10.0, 100.0, 1.0), 't': target(0.0)}
        )
        donors = donor_stage(buildings, {'t': 'd'})
        donors.add_asset(
            stray, Asset(stray, square(0.0, 0.0), {IMPUTATION_DONOR_KEY: 'd'})
        )
        with pytest.raises(ValueError, match='holds assets the stage lacks'):
            run(buildings, donors)

    def test_donor_outside_the_stage(self) -> None:
        """A donor the stage lacks raises."""
        buildings = inventory({'t': target(0.0)})
        with pytest.raises(ValueError, match='which the stage lacks'):
            run(buildings, donor_stage(buildings, {'t': 'gone'}))

    @pytest.mark.parametrize(
        ('n_pw', 'world'), [(2, None), (3, 3), (3, -1), (3, True)]
    )
    def test_worlds_refused(self, n_pw: int, world: Any) -> None:
        """Stages of different world counts, or a world outside them, raise."""
        buildings = inventory({'t': target(0.0)})
        buildings.n_pw = 3
        with pytest.raises(ValueError, match='world'):
            run(
                buildings,
                donor_stage(buildings, {'t': None}, n_pw=n_pw),
                world=world,
            )

    @pytest.mark.parametrize(
        ('features', 'error'),
        [
            ('OccupancyClass', TypeError),
            (('OccupancyClass', 'OccupancyClass', 'ReplacementCost'), ValueError),
            (('OccupancyClass', 'NumberOfStories'), ValueError),
        ],
    )
    def test_source_features_refused(self, features: Any, error: type) -> None:
        """A single string, a repeat or a missing value feature raises."""
        with pytest.raises(error):
            ReplacementCostImputer(source_features=features)

    @pytest.mark.parametrize(
        ('options', 'message'),
        [
            ({'location_scale_m': 0.0}, 'location_scale_m must be positive'),
            ({'area_weight': -1.0}, 'must not be negative'),
            ({'stories_weight': -1.0}, 'must not be negative'),
            ({'k': 0}, 'at least 1'),
        ],
    )
    def test_draw_options_refused(
        self, options: dict[str, float], message: str
    ) -> None:
        """A non-positive location scale, a negative weight or k 0 raises."""
        with pytest.raises(ValueError, match=message):
            ReplacementCostImputer(**options)
