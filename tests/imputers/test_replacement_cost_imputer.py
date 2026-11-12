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

from brails.aggregators import BlockGroupUnitBudget
from brails.aggregators.unit_budget.unit_budget import (
    AREA_FROM_GROSS_AREA,
    AREA_FROM_PLAN_AREA,
    AREA_FROM_STORIES,
    GEOID_COLUMN,
)
from brails.compilers.stage_freeze import StageFreeze
from brails.imputers import ObservedDonorImputer, ReplacementCostImputer
from brails.imputers.observed_donor_imputer.donor_records import (
    LOG_COLUMNS as DONOR_LOG_COLUMNS,
)
from brails.imputers.observed_donor_imputer.donor_records import (
    OUTCOME_DONOR,
    OUTCOME_NO_AREA,
    ROUND_DONOR,
    ROUND_REPAIR,
)
from brails.imputers.replacement_cost.replacement_cost_imputer import (
    IMPUTATION_DONOR_KEY,
    OUTCOME_IMPUTATION_DONOR,
    OUTCOME_NO_CLASS,
    OUTCOME_NO_DONOR_RATE,
    ReplacementCostResult,
    donor_stage_from_log,
)
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils.inventory_union import union_disjoint
from tests.imputers.donor_fixtures import asset, inventory, square

SQFT = 0.3048**2


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
            The eight codes: the given class, stories and value codes, and
            ``-`` for the other five features.
    """
    return f'{occupancy}-{stories}----{value}'


def nsi_building(
    east: float, value: float, gross: float, occupancy: str = 'COM1', **features: Any
) -> tuple[list, dict[str, Any]]:
    """
    Return a building whose class and value NSI states.

    Args:
        east (float):
            Meters east of the origin.
        value (float):
            Its replacement cost.
        gross (float):
            Its gross floor area in square feet.
        occupancy (str, optional):
            Its class. Defaults to ``'COM1'``.
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
    Return a building without a value, its class and stories imputed.

    Args:
        east (float):
            Meters east of the origin.
        occupancy (Any, optional):
            Its class. Defaults to ``'COM1'``.
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
    buildings: AssetInventory, donors: AssetInventory, world: int | None = None
) -> ReplacementCostResult:
    """
    Freeze a stage and its donor stage and impute it.

    Args:
        buildings (AssetInventory):
            The stage.
        donors (AssetInventory):
            The donor stage.
        world (int, optional):
            The world. Defaults to every world.

    Returns:
        ReplacementCostResult:
            The result.
    """
    freeze = StageFreeze()
    freeze.freeze('stage', buildings)
    freeze.freeze('donors', donors)
    return ReplacementCostImputer().impute_stage(
        freeze, 'stage', 'donors', world=world
    )


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
    Return the one log row of a target.

    Args:
        result (ReplacementCostResult):
            The result.
        asset_id (str):
            The target.

    Returns:
        dict[str, Any]:
            The row.
    """
    rows = result.log[result.log['asset_id'] == asset_id].to_dict('records')
    assert len(rows) == 1
    return rows[0]


def value_of(result: ReplacementCostResult, asset_id: str) -> Any:
    """
    Return a target's stored value, or None where it received none.

    Args:
        result (ReplacementCostResult):
            The result.
        asset_id (str):
            The target.

    Returns:
        Any:
            The stored value.
    """
    held = result.imputed.inventory.get(asset_id)
    return None if held is None else held.features['ReplacementCost']


class TestJointCopy:
    """A target takes its donor's rate, scaled to its own floor area."""

    def test_value_scaled_to_the_floor_area(self) -> None:
        """The value is the donor's rate times the target's gross floor area."""
        buildings = inventory(
            {'d': nsi_building(500.0, 450000.0, 1800.0), 't': target(0.0)}
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}))
        expected = 450000.0 / 1800.0 * plan_sqft(buildings, 't') * 2
        assert value_of(result, 't') == pytest.approx(expected, rel=1e-12)
        row = logged(result, 't')
        assert row['outcome'] == OUTCOME_IMPUTATION_DONOR
        assert row['donor_id'] == 'd'
        assert row['rate'] == 250.0
        assert row['area_source'] == AREA_FROM_STORIES
        assert math.isnan(row['distance_m'])

    def test_nsi_values_are_kept(self) -> None:
        """A value coded N or R is kept, whatever the code of the class."""
        buildings = inventory(
            {
                'n': nsi_building(0.0, 1.0, 1.0),
                'r': nsi_building(50.0, 2.0, 1.0, SourceString=source('N', 'R')),
                'i': nsi_building(80.0, 3.0, 1.0, SourceString=source('I', 'N')),
            }
        )
        result = run(buildings, donor_stage(buildings, {}))
        assert result.log.empty
        assert result.imputed.get_asset_ids() == []

    def test_donor_rate_over_a_nearer_building_and_another_class(self) -> None:
        """The named donor's rate is used over a nearer building of the class."""
        buildings = inventory(
            {
                'near': nsi_building(10.0, 100.0, 1.0),
                'far': nsi_building(1500.0, 300.0, 1.0, occupancy='RES3A'),
                't': target(0.0, GrossArea=10.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {'t': 'far'}))
        assert value_of(result, 't') == 3000.0

    def test_redrawn_donor_is_used(self) -> None:
        """Where validation re-drew, the re-draw's donor gives the rate."""
        buildings = inventory(
            {
                'first': nsi_building(10.0, 100.0, 1.0),
                'second': nsi_building(20.0, 200.0, 1.0),
                't': target(0.0, GrossArea=1.0),
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
        result = run(buildings, donor_stage_from_log(log, targets, n_worlds=1))
        assert value_of(result, 't') == 200.0


class TestFloorArea:
    """The floor area: gross area, else plan area times stories, else plan area."""

    @pytest.mark.parametrize('stories', [1, 4, None, 'NA'])
    def test_gross_area_wins(self, stories: Any) -> None:
        """A target's own gross area is used whatever its stories."""
        buildings = inventory(
            {
                'd': nsi_building(10.0, 100.0, 1.0),
                't': target(0.0, GrossArea=7.0, NumberOfStories=stories),
            }
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}))
        assert value_of(result, 't') == 700.0
        assert logged(result, 't')['area_source'] == AREA_FROM_GROSS_AREA

    @pytest.mark.parametrize('stories', [None, 0, -1, math.nan, True, 'two'])
    def test_plan_area_as_one_story(self, stories: Any) -> None:
        """Without a usable story count the plan area counts as one story."""
        buildings = inventory(
            {
                'd': nsi_building(10.0, 100.0, 1.0),
                't': target(0.0, NumberOfStories=stories, GrossArea=0.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}))
        row = logged(result, 't')
        assert row['area_source'] == AREA_FROM_PLAN_AREA
        assert row['floor_area_sqft'] == pytest.approx(plan_sqft(buildings, 't'))
        assert value_of(result, 't') == pytest.approx(100.0 * row['floor_area_sqft'])

    def test_no_area(self) -> None:
        """A target without a plan area or a gross area gets no value."""
        lon, lat = square(0.0, 0.0)[0]
        buildings = inventory({'d': nsi_building(10.0, 100.0, 1.0)})
        buildings.add_asset(
            't',
            Asset(
                't',
                [[lon, lat]],
                {'OccupancyClass': 'COM1', 'SourceString': source()},
            ),
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}))
        assert logged(result, 't')['outcome'] == OUTCOME_NO_AREA
        assert value_of(result, 't') is None


class TestUnitBudgetFloorArea:
    """The floor area is the gross floor area the unit budget reads."""

    def test_same_area_and_source(self) -> None:
        """
        The two agree on every footprint both measure.

        They differ on purpose where one gives no floor area: a footprint
        without area and without a gross area has none here and a plan area
        of zero in the budget, and unreadable coordinates raise in the
        budget even with a gross area, which this rule reads first.
        """
        records = {
            'gross': target(0.0, GrossArea=500.0, NumberOfStories=3),
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
        columns = [GEOID_COLUMN] + [
            f'B25024_{line:03d}{kind}' for line in range(2, 11) for kind in 'EM'
        ]
        budget = BlockGroupUnitBudget(
            pd.DataFrame(columns=columns), [], imputer=ObservedDonorImputer(seed=1)
        )
        sites = budget._sites(buildings)
        imputer = ReplacementCostImputer()
        for asset_id in records:
            area, source_name = imputer._floor_area(
                asset_id, buildings.inventory[asset_id]
            )
            site = sites[asset_id]
            assert source_name == site.gross_source, asset_id
            assert area == pytest.approx(site.gross_m2 / SQFT, rel=1e-12), asset_id


class TestOutcomes:
    """Targets that get no value are logged with the reason."""

    @pytest.mark.parametrize('occupancy', [None, 'NA', '', 3])
    def test_no_class(self, occupancy: Any) -> None:
        """A target without a class gets no_class whatever its area."""
        buildings = inventory(
            {
                'd': nsi_building(10.0, 100.0, 1.0),
                't': target(0.0, occupancy=occupancy, GrossArea=5.0),
            }
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}))
        row = logged(result, 't')
        assert row['outcome'] == OUTCOME_NO_CLASS
        assert row['area_source'] is None
        assert value_of(result, 't') is None

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
    def test_donor_without_a_rate(self, donor: dict[str, Any]) -> None:
        """A donor without a positive NSI value and gross area gives no rate."""
        buildings = inventory(
            {'d': nsi_building(10.0, 100.0, 1.0, **donor), 't': target(0.0)}
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}))
        row = logged(result, 't')
        assert row['outcome'] == OUTCOME_NO_DONOR_RATE
        assert row['donor_id'] == 'd'
        assert value_of(result, 't') is None

    def test_no_named_donor(self) -> None:
        """A target the donor stage lacks, or names no donor for, gets no rate."""
        buildings = inventory({'t': target(0.0), 'u': target(30.0)})
        result = run(buildings, donor_stage(buildings, {'u': [None]}))
        for asset_id in ('t', 'u'):
            row = logged(result, asset_id)
            assert row['outcome'] == OUTCOME_NO_DONOR_RATE
            assert row['donor_id'] is None

    def test_unreached_and_withheld_donors(self) -> None:
        """A donor NSI did not reach, or whose value was withheld, gives no rate."""
        buildings = inventory(
            {
                'unreached': target(10.0),
                'withheld': nsi_building(
                    20.0, 'NA', 1.0, SourceString=source('I', '-')
                ),
                't': target(0.0),
                'u': target(5.0),
            }
        )
        result = run(
            buildings, donor_stage(buildings, {'t': 'unreached', 'u': 'withheld'})
        )
        assert {logged(result, a)['outcome'] for a in ('t', 'u')} == {
            OUTCOME_NO_DONOR_RATE
        }


class TestProvenance:
    """After the merge the value carries the code of its source."""

    def test_eighth_character(self) -> None:
        """A donor value is coded I, NSI's N or R, and a target without one -."""
        records = {
            'n': asset(0.0, 0.0),
            'r': asset(30.0, 0.0),
            't': asset(60.0, 0.0),
            'x': asset(90.0, 0.0),
        }
        footprint = inventory({a: (c, {}) for a, (c, _) in records.items()})
        nsi = inventory(
            {
                'n': (
                    records['n'][0],
                    {
                        'OccupancyClass': 'COM1',
                        'ReplacementCost': 900.0,
                        'GrossArea': 3.0,
                    },
                ),
                'r': (records['r'][0], {'OccupancyClass': 'COM1', 'GrossArea': 2.0}),
            }
        )
        derived = inventory({'r': (records['r'][0], {'ReplacementCost': 500.0})})
        # x is of a class no building with a value holds
        filled = inventory(
            {
                a: (records[a][0], {'OccupancyClass': cls, 'NumberOfStories': 1})
                for a, cls in (('t', 'COM1'), ('x', 'IND1'))
            }
        )
        precedence = ['footprint', 'nsi', 'derived', 'imputed']

        def merged(imputed: AssetInventory) -> AssetInventory:
            """
            Merge the sources with the given imputed source.

            Args:
                imputed (AssetInventory):
                    The imputed source.

            Returns:
                AssetInventory:
                    The merged inventory.
            """
            return footprint.merge(
                {'nsi': nsi, 'derived': derived, 'imputed': imputed},
                precedence,
                name='footprint',
            )

        buildings = merged(filled)
        result = run(buildings, donor_stage(buildings, {'t': 'n', 'x': None}))
        final = merged(union_disjoint(filled, result.imputed))
        codes = {a: final.inventory[a].features['SourceString'][7] for a in records}
        assert codes == {'n': 'N', 'r': 'R', 't': 'I', 'x': '-'}
        assert final.inventory['n'].features['ReplacementCost'] == 900.0
        assert final.inventory['r'].features['ReplacementCost'] == 500.0
        assert final.inventory['t'].features['ReplacementCost'] == pytest.approx(
            300.0 * plan_sqft(buildings, 't')
        )


class TestWorlds:
    """Each world reads its own realization of both stages."""

    def three_worlds(self) -> tuple[AssetInventory, AssetInventory]:
        """
        Return a three-world stage and donor stage.

        Returns:
            tuple[AssetInventory, AssetInventory]:
                The stage, whose target's stories and class differ by
                world, and the donor stage, whose donor does.
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
        return buildings, donor_stage(buildings, {'t': ['a', 'b', 'b']}, n_pw=3)

    def test_per_world_values(self) -> None:
        """Every world is computed, and a world without a value holds None."""
        buildings, donors = self.three_worlds()
        result = run(buildings, donors)
        plan = plan_sqft(buildings, 't')
        stored = value_of(result, 't')
        assert stored[0] == pytest.approx(100.0 * plan)
        assert stored[1] is None
        assert stored[2] == pytest.approx(900.0 * plan)
        assert result.imputed.n_pw == 3
        assert list(result.log['outcome']) == [
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
        assert one.imputed.n_pw == 1
        for result in (one, scalar):
            assert value_of(result, 't') == value_of(whole, 't')[2]
            pd.testing.assert_frame_equal(
                result.log,
                whole.log[whole.log['world'] == 2].reset_index(drop=True),
            )

    def test_agreeing_worlds_collapse(self) -> None:
        """A value equal in every world is stored as a scalar."""
        buildings = inventory(
            {'d': nsi_building(10.0, 100.0, 1.0), 't': target(0.0, GrossArea=2.0)}
        )
        result = run(buildings, donor_stage(buildings, {'t': 'd'}, n_pw=4))
        assert value_of(result, 't') == 200.0
        assert result.imputed.n_pw == 4
        assert len(result.log) == 4


class TestStagesAndArguments:
    """What impute_stage reads and refuses."""

    def test_stages_unchanged(self) -> None:
        """The manifest passes after a run, with the digests of the freeze."""
        buildings = inventory(
            {
                'd': nsi_building(10.0, 100.0, 1.0),
                't': target(0.0),
                'u': target(30.0, GrossArea=1.0),
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
        ('change', 'message'),
        [
            ({'SourceString': None}, 'no source string'),
            ({'SourceString': 'NNN'}, 'no source string'),
        ],
    )
    def test_donor_source_string(self, change: dict[str, Any], message: str) -> None:
        """A donor's missing or short source string raises."""
        buildings = inventory(
            {'d': nsi_building(10.0, 100.0, 1.0, **change), 't': target(0.0)}
        )
        with pytest.raises(ValueError, match=message):
            run(buildings, donor_stage(buildings, {'t': 'd'}))

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
