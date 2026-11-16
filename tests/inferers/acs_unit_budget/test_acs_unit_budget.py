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

"""Tests for the ACS unit budget."""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from brails.constants import (
    BLOCK_GROUP_KEY,
    GROSS_AREA_KEY,
    OCCUPANCY_KEY,
    SOURCE_STRING_KEY,
    STORIES_KEY,
    UNITS_KEY,
)
from brails.inferers import ACSUnitBudget
from brails.inferers.acs_unit_budget.acs_unit_budget import (
    B25024_BINS,
    BIN_CLASSES,
    GAP_BIN_WITHOUT_STRUCTURE,
    GAP_CANDIDATE_SHORTFALL,
    GAP_MOBILE_HOME_SHORTFALL,
    GAP_NO_ACS,
    GAP_STRUCTURES_WITHOUT_UNITS,
    GAP_UNITS_CLIPPED,
    GAP_UNITS_OVER_CAPACITY,
    MOBILE_HOME_LABEL,
    OUTCOME_DROPPED,
    OUTCOME_DWELLING,
    OUTCOME_NO_ACS,
    OUTCOME_NONRESIDENTIAL,
    PURPOSE_UNIT_BUDGET,
    UnitBudgetResult,
)
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils import Importer
from brails.utils.source_updates import WITHHELD_COLUMNS, apply_to_sources
from brails.utils.unit_count_rules import UNIT_MIDPOINTS, count_band
from tests.imputers.donor_fixtures import (
    BG_A,
    BG_B,
    OTHER_SEED,
    SEED,
    asset,
    features_of,
    imputer,
    inventory,
    lonlat,
    world_value,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

OCC = OCCUPANCY_KEY
UNITS = UNITS_KEY
STORIES = STORIES_KEY
MH_PARK = (-500.0, 0.0)
WORLDS = 3
# The reached footprint of the main fixture the mobile-home rule may
# reclassify, with the label of its source.
RELABEL = {'rp': 'nsi'}
# The fixtures' block groups often hold more structures than candidates,
# and the tests of the warning catch it themselves.
pytestmark = pytest.mark.filterwarnings(
    r'ignore:In \d+ of the block groups:UserWarning'
)


def table(rows: dict[str, dict[str, tuple[float, float]]]) -> pd.DataFrame:
    """
    Build a B25024 table.

    Args:
        rows (dict[str, dict[str, tuple[float, float]]]):
            Per GEOID, the estimate and margin per table line, zero where
            a line is not given.

    Returns:
        pandas.DataFrame:
            The table with every line's estimate and margin column.
    """
    records = []
    for geoid, lines in rows.items():
        record: dict[str, Any] = {'geoid': geoid}
        for number in range(1, 12):
            line = f'{number:03d}'
            estimate, margin = lines.get(line, (0, 0))
            record[f'B25024_{line}E'] = estimate
            record[f'B25024_{line}M'] = margin
        records.append(record)
    columns = ['geoid'] + [
        f'B25024_{n:03d}{kind}' for n in range(1, 12) for kind in 'EM'
    ]
    return pd.DataFrame(records, columns=columns)


def in_bg(block_group: str = BG_A, **features: Any) -> dict[str, Any]:
    """
    Return features with a block group.

    Args:
        block_group (str, optional):
            The GEOID. Defaults to the first fixture block group.
        **features (Any):
            Other features.

    Returns:
        dict[str, Any]:
            The features.
    """
    return {BLOCK_GROUP_KEY: block_group, **features}


def house(east: float, north: float, side: float = 10.0, **extra: Any) -> tuple:
    """
    Return a reached single-family footprint.

    Args:
        east (float):
            Meters east.
        north (float):
            Meters north.
        side (float, optional):
            Side in meters.
        **extra (Any):
            Feature overrides.

    Returns:
        tuple:
            Coordinates and features.
    """
    features = {OCC: 'RES1', UNITS: 1, STORIES: 1, **extra}
    return asset(east, north, side, **in_bg(**features))


def empty(east: float, north: float, side: float = 10.0, bg: str = BG_A) -> tuple:
    """
    Return an unreached footprint.

    Args:
        east (float):
            Meters east.
        north (float):
            Meters north.
        side (float, optional):
            Side in meters. A 10 m side gives 1,076 sq ft and an 8 m side
            689 sq ft.
        bg (str, optional):
            The block group.

    Returns:
        tuple:
            Coordinates and features.
    """
    return asset(east, north, side, **in_bg(bg))


def scene() -> tuple[pd.DataFrame, list, AssetInventory, AssetInventory]:
    """
    Build the main fixture: homes in the west, shops in the east, a mobile-home park.

    The mobile-home park point lies at ``MH_PARK``. Four houses, a fifth
    house near the mobile-home park (``rp``) and five shops in the east are
    reached. The targets are three footprints near or beyond the
    mobile-home park, eight footprints of growing size in the west, and two
    small and one large footprint among the shops.

    Returns:
        tuple[pandas.DataFrame, list, AssetInventory, AssetInventory]:
            The table, the mobile-home park points, the targets and the
            reached footprints.
    """
    reached = inventory(
        {
            **{f'r{i}': house(0, 40 * i) for i in range(4)},
            # reached commercial footprints
            **{
                f'c{i}': asset(
                    1500, 40 * i, 20.0, **in_bg(**{OCC: 'COM1', STORIES: 1})
                )
                for i in range(5)
            },
            'rp': house(-450, 0),
        }
    )
    targets = inventory(
        {
            'm1': empty(-480, 20),
            'm2': empty(-420, -80),
            'm3': empty(-150, 0),
            **{f't{i}': empty(60, 40 * i, 10 + i) for i in range(8)},
            's1': empty(1450, 0, 8),
            's2': empty(1450, 60, 8),
            'b1': empty(1550, 100, 30),
        }
    )
    rows = table({BG_A: {'002': (8, 3), '006': (14, 5), '010': (2, 1)}})
    return rows, [lonlat(*MH_PARK)], targets, reached


def budget(rows: pd.DataFrame, mh_parks: list, **kwargs: Any) -> ACSUnitBudget:
    """
    Build a budget on the fixture imputer and seed.

    Args:
        rows (pandas.DataFrame):
            The table.
        mh_parks (list):
            The mobile-home park points.
        **kwargs (Any):
            Constructor overrides.

    Returns:
        ACSUnitBudget:
            The budget.
    """
    options = {'imputer': imputer(), 'seed': SEED}
    options.update(kwargs)
    return ACSUnitBudget(rows, mh_parks, **options)


def run(
    n_worlds: int = WORLDS,
    relabel_sources: Mapping[str, str] | None = RELABEL,
    **kwargs: Any,
) -> UnitBudgetResult:
    """
    Allocate the main fixture.

    Args:
        n_worlds (int, optional):
            The number of worlds.
        relabel_sources (Mapping[str, str] | None, optional):
            The reached footprints the mobile-home rule may reclassify.
            Defaults to ``RELABEL``.
        **kwargs (Any):
            Constructor overrides.

    Returns:
        UnitBudgetResult:
            The allocation.
    """
    rows, mh_parks, targets, reached = scene()
    return budget(rows, mh_parks, **kwargs).allocate(
        targets, reached, n_worlds=n_worlds, relabel_sources=relabel_sources
    )


def outcomes(result: UnitBudgetResult) -> dict[str, str]:
    """
    Return each logged footprint's outcome.

    Args:
        result (UnitBudgetResult):
            An allocation.

    Returns:
        dict[str, str]:
            Asset id to outcome.
    """
    return dict(zip(result.log['asset_id'], result.log['outcome']))


def gap_kinds(result: UnitBudgetResult) -> dict[tuple, tuple]:
    """
    Return the gaps by block group, bin and kind.

    Args:
        result (UnitBudgetResult):
            An allocation.

    Returns:
        dict[tuple, tuple]:
            ``(block_group, bin, kind)`` to ``(count, units)``.
    """
    return {
        (g.block_group, g.bin, g.kind): (g.count, g.units)
        for g in result.gaps.itertuples(index=False)
    }


def units_by_bin(result: UnitBudgetResult, world: int) -> dict[str, int]:
    """
    Sum the dwelling units the budget placed per bin in one world.

    Args:
        result (UnitBudgetResult):
            An allocation.
        world (int):
            The world.

    Returns:
        dict[str, int]:
            Units per bin label.
    """
    bin_of = {c: b for b, c in BIN_CLASSES.items()}
    totals = dict.fromkeys(B25024_BINS, 0)
    for features in features_of(result.imputed).values():
        occupancy = world_value(features[OCC], world)
        if occupancy in bin_of and UNITS in features:
            totals[bin_of[occupancy]] += world_value(features[UNITS], world)
    return totals


def random_scene(rng: random.Random) -> tuple:
    """
    Build a random block group with reached and unreached footprints.

    Args:
        rng (random.Random):
            The generator.

    Returns:
        tuple:
            The table, no mobile-home park points, the targets and the
            reached footprints.
    """
    reached = {}
    for i in range(rng.randint(3, 12)):
        occupancy, units = rng.choice(
            [('RES1', 1), ('RES3A', 2), ('RES3C', 6), ('RES3E', 30), ('COM1', None)]
        )
        extra = {OCC: occupancy, STORIES: 1}
        if units is not None:
            extra[UNITS] = units
        reached[f'r{i}'] = asset(
            rng.uniform(0, 900), rng.uniform(0, 900), **in_bg(**extra)
        )
    targets = {
        f't{i}': empty(rng.uniform(0, 900), rng.uniform(0, 900), rng.uniform(6, 25))
        for i in range(rng.randint(1, 30))
    }
    lines = {
        line: (rng.randint(0, 60), rng.randint(0, 10))
        for parts in B25024_BINS.values()
        for line in parts
    }
    return table({BG_A: lines}), [], inventory(targets), inventory(reached)


class TestBudget:
    """The units placed never exceed a bin's residual nor a dwelling's capacity."""

    def test_main_fixture(self) -> None:
        """No bin receives more than its whole residual in any world."""
        result = run()
        for world in range(WORLDS):
            units = units_by_bin(result, world)
            # Bin 1 holds 8 less the four reached houses, since rp became a
            # mobile home.
            assert units['1'] <= 4
            assert units['5-9'] <= 14
            assert sum(units.values()) == 4 + 14

    @pytest.mark.parametrize('per_bin', [False, True])
    @pytest.mark.parametrize('seed', range(12))
    def test_random_block_groups(self, seed: int, *, per_bin: bool) -> None:
        """In random block groups no bin and no dwelling receives too much."""
        rng = random.Random(seed)
        rows, mh_parks, targets, reached = random_scene(rng)
        result = budget(rows, mh_parks, seed=seed, per_bin=per_bin).allocate(
            targets, reached, n_worlds=WORLDS
        )
        bin_of = {c: b for b, c in BIN_CLASSES.items()}
        reached_units = dict.fromkeys(B25024_BINS, 0.0)
        for x in reached.inventory.values():
            if x.features[OCC] in bin_of:
                reached_units[bin_of[x.features[OCC]]] += x.features[UNITS]
        estimates = {
            label: sum(rows.iloc[0][f'B25024_{line}E'] for line in lines)
            for label, lines in B25024_BINS.items()
        }
        deficit = max(0.0, sum(estimates.values()) - sum(reached_units.values()))
        capacity = dict(zip(result.log['asset_id'], result.log['capacity']))
        for world in range(WORLDS):
            placed = units_by_bin(result, world)
            for label in B25024_BINS:
                bound = math.floor(max(0.0, estimates[label] - reached_units[label]))
                assert placed[label] <= bound
            if not per_bin:
                assert sum(placed.values()) <= deficit
            for asset_id, features in features_of(result.imputed).items():
                assert world_value(features[UNITS], world) <= capacity[asset_id]
            assert placed == units_by_bin(result, 0)


class TestTotalDeficit:
    """By default the bins share the block group's total deficit."""

    def scene(self) -> tuple:
        """
        Build a block group whose reached units meet the census total.

        Reached duplexes hold more two-unit homes than the census counts,
        and fewer single-family homes are reached than it counts.

        Returns:
            tuple:
                The table, the targets and the reached footprints.
        """
        rows = table({BG_A: {'002': (6, 2), '004': (4, 2)}})
        reached = inventory(
            {
                **{f'r{i}': house(0, 40 * i) for i in range(2)},
                **{
                    f'd{i}': house(0, 40 * i + 200, **{OCC: 'RES3A', UNITS: 2})
                    for i in range(4)
                },
            }
        )
        targets = inventory({f't{i}': empty(60, 40 * i) for i in range(6)})
        return rows, targets, reached

    def test_total_met_places_nothing(self) -> None:
        """Where reached units meet the total, no dwelling is made."""
        rows, targets, reached = self.scene()
        result = budget(rows, []).allocate(targets, reached)
        assert features_of(result.imputed) == {}
        total = result.diagnostic.set_index('bin').loc['total']
        assert (total['estimate'], total['reached_units']) == (10.0, 10.0)

    def test_per_bin_option(self) -> None:
        """With per_bin, the one-unit bin's residual stands on its own."""
        rows, targets, reached = self.scene()
        result = budget(rows, [], per_bin=True).allocate(targets, reached)
        assert units_by_bin(result, 0)['1'] == 4

    def test_shared_in_proportion(self) -> None:
        """A deficit below the bins' sum is shared by their residuals."""
        rows = table({BG_A: {'002': (6, 2), '006': (12, 2)}})
        reached = inventory(
            {f'd{i}': house(0, 40 * i, **{OCC: 'RES3A', UNITS: 2}) for i in range(3)}
        )
        # 16 m sides give room for 9 units.
        targets = inventory({f't{i}': empty(60, 40 * i, 16) for i in range(8)})
        result = budget(rows, []).allocate(targets, reached)
        # Deficit 18 - 6 = 12 shared by residuals 6 and 12: 4 and 8 units.
        assert units_by_bin(result, 0)['1'] == 4
        assert units_by_bin(result, 0)['5-9'] == 8

    def test_fractional_split_loses_no_unit(self) -> None:
        """An odd deficit split between two bins keeps every whole unit."""
        rows = table({BG_A: {'002': (5, 1), '004': (5, 1)}})
        reached = inventory({'q': house(0, 0, **{OCC: 'RES3B', UNITS: 3})})
        targets = inventory({f't{i}': empty(60, 40 * i) for i in range(8)})
        result = budget(rows, []).allocate(targets, reached)
        # Deficit 10 - 3 = 7 over whole residuals 5 and 5: 4 and 3 units.
        diagnostic = result.diagnostic.set_index('bin')['generated_units']
        unplaced = result.gaps['units'].fillna(0).sum()
        assert diagnostic['1'] == 4
        assert diagnostic['total'] + unplaced == 7


class TestCandidacy:
    """Candidacy is decided once, and class and units vary by world."""

    def test_same_dwellings_every_world(self) -> None:
        """Every world holds the same dwellings, and their bins differ."""
        result = run(n_worlds=6)
        log = result.log
        dwellings = set(log.loc[log['outcome'] == OUTCOME_DWELLING, 'asset_id'])
        features = features_of(result.imputed)
        for world in range(6):
            in_world = {
                a
                for a, f in features.items()
                if world_value(f[OCC], world) in BIN_CLASSES.values()
            }
            assert in_world == dwellings
        varied = [a for a in dwellings if isinstance(features[a][OCC], list)]
        assert varied

    def test_world_zero_independent_of_world_count(self) -> None:
        """World 0's allocation does not depend on how many worlds are drawn."""
        one = features_of(run(n_worlds=1).imputed)
        six = features_of(run(n_worlds=6).imputed)
        assert one.keys() == six.keys()
        for asset_id, features in one.items():
            for key, value in features.items():
                assert world_value(six[asset_id][key], 0) == value

    def test_class_is_band_of_units(self) -> None:
        """Every dwelling's class is the count band of its units."""
        result = run(n_worlds=6)
        for asset_id, features in features_of(result.imputed).items():
            if not asset_id.startswith('t'):
                continue
            for world in range(6):
                units = world_value(features[UNITS], world)
                assert count_band(units) == world_value(features[OCC], world)

    def test_rank_by_probability_then_area(self) -> None:
        """Residential neighbors rank first, then larger footprints."""
        log = run().log.set_index('asset_id')
        assert log.loc['t7', 'rank'] == 1
        assert log.loc['t6', 'rank'] == 2
        assert log.loc['b1', 'probability'] == 0.0
        assert log.loc['b1', 'rank'] > log.loc['t0', 'rank']


class TestDropped:
    """Small unchosen footprints are dropped unless near a mobile-home park."""

    def test_small_dropped_and_logged(self) -> None:
        """An unchosen footprint under 1,000 sq ft is dropped and logged."""
        result = run()
        assert result.dropped == ('s1', 's2')
        assert outcomes(result)['s1'] == OUTCOME_DROPPED
        assert 's1' not in result.residential

    def test_mh_park_exempts(self) -> None:
        """A small footprint within 300 m of a mobile-home park point is kept."""
        rows = table({BG_A: {}})
        targets = inventory({'near': empty(-300, 0, 8), 'far': empty(300, 0, 8)})
        reached = inventory({'r0': house(0, 0)})
        result = budget(rows, [lonlat(*MH_PARK)]).allocate(targets, reached)
        assert result.dropped == ('far',)
        assert result.nonresidential == ('near',)

    def test_at_threshold_kept(self) -> None:
        """A footprint at the threshold is passed on, one below it dropped."""
        rows = table({BG_A: {}})
        targets = inventory({'a': empty(300, 0, 8), 'b': empty(300, 60, 8)})
        reached = inventory({'r0': house(0, 0)})
        area = budget(rows, []).allocate(targets, reached).log.set_index('asset_id')
        at = float(area.loc['a', 'plan_area_sqft'])
        result = budget(rows, [], threshold_sqft=at).allocate(targets, reached)
        assert 'a' in result.nonresidential
        above = budget(rows, [], threshold_sqft=at + 1e-6).allocate(targets, reached)
        assert 'a' in above.dropped

    def test_no_area_dropped(self) -> None:
        """A target whose rings enclose no area is dropped."""
        rows = table({BG_A: {'002': (3, 1)}})
        flat = [[-122.0, 38.0]] * 4
        targets = inventory({'flat': (flat, in_bg()), 't': empty(0, 0)})
        result = budget(rows, []).allocate(targets, inventory({'r0': house(0, 50)}))
        assert result.dropped == ('flat',)

    def test_threshold_argument(self) -> None:
        """A threshold below the footprint's area keeps it."""
        result = run(threshold_sqft=600.0)
        assert result.dropped == ()


class TestNonResidential:
    """Larger unchosen footprints go to the non-residential stage."""

    def test_large_unchosen(self) -> None:
        """An unchosen footprint at or above the threshold is passed on."""
        result = run()
        assert 'b1' in result.nonresidential
        assert result.residential['b1'] is False
        assert outcomes(result)['b1'] == OUTCOME_NONRESIDENTIAL


class TestGaps:
    """Every unplaced part of a budget is reported."""

    def test_shortfall_and_clipping(self) -> None:
        """Too few candidates leave structures unplaced and units clipped."""
        rows = table({BG_A: {'002': (10, 1)}})
        targets = inventory({'t0': empty(0, 0)})
        with pytest.warns(UserWarning, match='9 structures found no footprint'):
            result = budget(rows, []).allocate(targets, inventory({}))
        gaps = gap_kinds(result)
        assert gaps[(BG_A, None, GAP_CANDIDATE_SHORTFALL)][0] == 9
        assert gaps[(BG_A, '1', GAP_UNITS_CLIPPED)][1] == 9
        assert features_of(result.imputed)['t0'] == {OCC: 'RES1', UNITS: 1}

    def test_bin_without_structure(self) -> None:
        """A residual too small for a structure is reported."""
        rows = table({BG_A: {'002': (1, 1), '009': (30, 5)}})
        targets = inventory({'t0': empty(0, 0), 't1': empty(50, 0)})
        result = budget(rows, []).allocate(targets, inventory({}))
        assert gap_kinds(result)[(BG_A, '50+', GAP_BIN_WITHOUT_STRUCTURE)] == (0, 30)

    def test_structure_without_units(self) -> None:
        """A structure the bin's whole residual cannot fill is removed."""
        rows = table({BG_A: {'008': (19, 5)}})
        targets = inventory({'t0': empty(0, 0)})
        result = budget(rows, []).allocate(targets, inventory({}))
        gaps = gap_kinds(result)
        assert gaps[(BG_A, '20-49', GAP_STRUCTURES_WITHOUT_UNITS)][0] == 1
        assert features_of(result.imputed) == {}

    def test_mobile_home_shortfall(self) -> None:
        """Mobile homes without eligible footprints are reported."""
        rows = table({BG_A: {'010': (3, 1)}})
        targets = inventory({'m': empty(-480, 0)})
        result = budget(rows, [lonlat(*MH_PARK)]).allocate(targets, inventory({}))
        gaps = gap_kinds(result)
        assert gaps[(BG_A, MOBILE_HOME_LABEL, GAP_MOBILE_HOME_SHORTFALL)] == (2, 2)


class TestMobileHomes:
    """Mobile homes go to footprints near mobile-home park points, nearest first."""

    def test_nearest_within_radius(self) -> None:
        """The residual goes to the nearest eligible footprints."""
        result = run()
        features = features_of(result.imputed)
        assert features['m1'] == {OCC: 'RES2', UNITS: 1}
        assert features['rp'] == {OCC: 'RES2'}
        assert features.get('m2', {}).get(OCC) != 'RES2'

    @pytest.mark.parametrize(('estimate', 'expected'), [(9, {'a', 'b'}), (1, {'a'})])
    def test_never_beyond_radius_or_residual(
        self, estimate: int, expected: set
    ) -> None:
        """No footprint beyond 300 m becomes RES2, and no more than the residual."""
        rows = table({BG_A: {'010': (estimate, 1)}})
        targets = inventory(
            {'a': empty(-480, 0), 'b': empty(-300, 50), 'c': empty(-100, 0)}
        )
        result = budget(rows, [lonlat(*MH_PARK)]).allocate(targets, inventory({}))
        mobile = {
            a for a, f in features_of(result.imputed).items() if f.get(OCC) == 'RES2'
        }
        assert mobile == expected

    def test_reached_res2_reduces_residual(self) -> None:
        """Reached RES2 units count against the mobile-home estimate."""
        rows = table({BG_A: {'010': (2, 1)}})
        targets = inventory({'a': empty(-480, 0), 'b': empty(-470, 0)})
        reached = inventory({'p': house(-490, 30, **{OCC: 'RES2'})})
        result = budget(rows, [lonlat(*MH_PARK)]).allocate(targets, reached)
        mobile = {
            a for a, f in features_of(result.imputed).items() if f.get(OCC) == 'RES2'
        }
        assert mobile == {'a'}


class TestRelabel:
    """Reached RES1 footprints near mobile-home parks may become mobile homes."""

    def test_withheld_and_coded_imputed(self) -> None:
        """The source's class is withheld, and the merge codes RES2 imputed."""
        rows, mh_parks, targets, reached = scene()
        result = budget(rows, mh_parks).allocate(
            targets, reached, relabel_sources=RELABEL
        )
        assert list(result.withheld.columns) == list(WITHHELD_COLUMNS)
        assert result.withheld.to_dict('records') == [
            {'asset_id': 'rp', 'feature': OCC, 'source': 'nsi', 'value': 'RES1'}
        ]
        spine = AssetInventory()
        for asset_id, x in [*reached.inventory.items(), *targets.inventory.items()]:
            spine.add_asset(asset_id, Asset(asset_id, deepcopy(x.coordinates), {}))
        precedence = ['footprint', 'nsi', 'imputed']
        sources = apply_to_sources(
            {'footprint': spine, 'nsi': reached, 'imputed': result.imputed},
            {},
            result.withheld,
            precedence=precedence,
        )
        merged = spine.merge(
            {k: v for k, v in sources.items() if k != 'footprint'},
            precedence,
            name='footprint',
        )
        rp = merged.inventory['rp'].features
        assert rp[OCC] == 'RES2'
        assert rp[SOURCE_STRING_KEY][:2] == 'IN'
        assert reached.inventory['rp'].features[OCC] == 'RES1'

    def test_without_relabel_sources(self) -> None:
        """Without relabel_sources no reached footprint changes."""
        result = run(relabel_sources=None)
        assert result.withheld.empty
        assert 'rp' not in features_of(result.imputed)
        assert features_of(result.imputed)['m2'] == {OCC: 'RES2', UNITS: 1}

    def test_fully_reached_block_group(self) -> None:
        """A block group without targets still places its mobile homes."""
        rows = table({BG_A: {'002': (1, 1), '010': (1, 1)}})
        reached = inventory({'rp': house(-450, 0)})
        result = budget(rows, [lonlat(*MH_PARK)]).allocate(
            inventory({}), reached, relabel_sources=RELABEL
        )
        assert result.withheld['asset_id'].tolist() == ['rp']
        assert set(result.diagnostic['bin']) >= {'1', 'total', 'mobile_home'}

    def test_relabel_sources(self) -> None:
        """Only the named footprints are relabeled, under their source label."""
        rows, mh_parks, targets, reached = scene()
        allocate = budget(rows, mh_parks).allocate
        named = allocate(targets, reached, relabel_sources={'rp': 'derived'})
        assert named.withheld['source'].tolist() == ['derived']
        other = allocate(targets, reached, relabel_sources={'r0': 'nsi'})
        assert other.withheld.empty


class TestDiagnostic:
    """The diagnostic flags bins and changes nothing."""

    def test_flag_and_no_repair(self) -> None:
        """Reached units above the upper bound are flagged, nothing else changes."""
        rows = table({BG_A: {'002': (2, 1), '006': (14, 2)}})
        reached = inventory({f'r{i}': house(0, 40 * i) for i in range(5)})
        targets = inventory({f't{i}': empty(60, 40 * i) for i in range(4)})
        flagged = budget(rows, []).allocate(targets, reached, n_worlds=2)
        wide = budget(rows, [], moe_factor=1000.0).allocate(
            targets, reached, n_worlds=2
        )
        diagnostic = flagged.diagnostic.set_index(['world', 'bin'])
        assert diagnostic.loc[(0, '1'), 'outside']
        assert diagnostic.loc[(1, '1'), 'total_units'] == 5
        assert not wide.diagnostic['outside'].any()
        assert features_of(flagged.imputed) == features_of(wide.imputed)
        assert flagged.dropped == wide.dropped


class TestNoAcs:
    """A block group without B25024 counts has no budget."""

    def test_missing_row(self) -> None:
        """Its targets are logged and keep no budget value."""
        rows = table({BG_A: {'002': (1, 1)}})
        targets = inventory({'t0': empty(0, 0), 'x': empty(100, 0, 8, bg=BG_B)})
        reached = inventory({'r0': house(0, 50)})
        result = budget(rows, []).allocate(targets, reached)
        assert outcomes(result)['x'] == OUTCOME_NO_ACS
        assert 'x' not in features_of(result.imputed)
        assert 'x' not in result.residential
        assert gap_kinds(result)[(BG_B, None, GAP_NO_ACS)][0] == 1

    def test_negative_estimate(self) -> None:
        """A negative estimate counts as a missing row."""
        rows = table({BG_A: {'002': (-666666666, -222222222)}})
        targets = inventory({'t0': empty(0, 0)})
        result = budget(rows, []).allocate(targets, inventory({'r0': house(0, 50)}))
        assert outcomes(result)['t0'] == OUTCOME_NO_ACS

    def test_drop_without_acs(self) -> None:
        """With the option, a small, likely non-residential target is dropped."""
        rows = table({})
        reached = inventory(
            {
                # reached commercial footprints
                **{
                    f'c{i}': asset(
                        0, 40 * i, 20.0, **in_bg(**{OCC: 'COM1', STORIES: 1})
                    )
                    for i in range(5)
                },
                'h': house(900, 0),
            }
        )
        targets = inventory({'s': empty(30, 0, 8), 'h2': empty(930, 0, 8)})
        kept = budget(rows, []).allocate(targets, reached)
        assert kept.dropped == ()
        dropped = budget(rows, [], drop_without_acs=True).allocate(targets, reached)
        assert dropped.dropped == ('s',)


class TestReading:
    """The table is read as the Census publishes it."""

    def test_margins(self) -> None:
        """A negative margin reads zero, and bin 1 sums two margins in quadrature."""
        rows = table({BG_A: {'002': (3, 3), '003': (1, 4), '006': (7, -555555555)}})
        result = budget(rows, []).allocate(
            inventory({'t0': empty(0, 0)}), inventory({})
        )
        diagnostic = result.diagnostic.set_index('bin')
        assert diagnostic.loc['1', 'margin'] == 5.0
        assert diagnostic.loc['1', 'estimate'] == 4.0
        assert diagnostic.loc['5-9', 'margin'] == 0.0

    def test_bad_tables(self) -> None:
        """A missing column or a repeated GEOID raises."""
        rows = table({BG_A: {}})
        with pytest.raises(ValueError, match='lacks'):
            budget(rows.drop(columns=['B25024_010M']), [])
        with pytest.raises(ValueError, match='repeats'):
            budget(pd.concat([rows, rows]), [])
        numeric = rows.assign(geoid=int(BG_A))
        with pytest.raises(ValueError, match='dtype=str'):
            budget(numeric, [])
        with pytest.raises(ValueError, match='12 digits'):
            budget(rows.assign(geoid=BG_A[1:]), [])

    def test_unreadable_footprint(self) -> None:
        """A footprint whose coordinates cannot be read raises."""
        targets = inventory({'bad': ([['x', 'y']], in_bg())})
        with pytest.raises(ValueError, match='bad'):
            budget(table({BG_A: {}}), []).allocate(targets, inventory({}))

    def test_bins_and_classes(self) -> None:
        """Each bin's class holds every unit count of the bin."""
        for label, occupancy in BIN_CLASSES.items():
            low, _, high = label.rstrip('+').partition('-')
            assert count_band(int(low)) == occupancy
            if high:
                assert count_band(int(high)) == occupancy
                assert count_band(int(high) + 1) != occupancy
        assert set(BIN_CLASSES) == set(B25024_BINS)
        assert set(BIN_CLASSES.values()) <= set(UNIT_MIDPOINTS)


GOLDEN = {
    't2': ['RES1', 'RES1', 'RES1'],
    't3': ['RES1', 'RES1', 'RES1'],
    't4': ['RES3C', 'RES1', 'RES3C'],
    't5': ['RES1', 'RES1', 'RES1'],
    't6': ['RES1', 'RES3C', 'RES3C'],
    't7': ['RES3C', 'RES3C', 'RES1'],
}


class TestKeyedAssignment:
    """The per-world assignment is keyed, ordered and local."""

    def test_golden(self) -> None:
        """The classes of the dwellings at the fixture seed are pinned."""
        features = features_of(run().imputed)
        observed = {
            a: [world_value(features[a][OCC], w) for w in range(WORLDS)]
            for a in sorted(features)
            if a.startswith('t')
        }
        assert observed == GOLDEN

    def test_purpose_label(self) -> None:
        """The per-world order's purpose label is pinned."""
        assert PURPOSE_UNIT_BUDGET == 'unit_budget'

    def test_seed_sensitivity(self) -> None:
        """Another seed moves at least one dwelling's class."""
        assert features_of(run(seed=OTHER_SEED).imputed) != features_of(
            run().imputed
        )

    def test_reordering(self) -> None:
        """The insertion order of either inventory changes nothing."""
        rows, mh_parks, targets, reached = scene()
        first = budget(rows, mh_parks).allocate(targets, reached, n_worlds=WORLDS)
        reverse_targets = inventory(
            {a: (x.coordinates, x.features) for a, x in targets.inventory.items()},
            order=list(reversed(targets.inventory)),
        )
        reverse_reached = inventory(
            {a: (x.coordinates, x.features) for a, x in reached.inventory.items()},
            order=list(reversed(reached.inventory)),
        )
        second = budget(rows, mh_parks).allocate(
            reverse_targets, reverse_reached, n_worlds=WORLDS
        )
        assert features_of(first.imputed) == features_of(second.imputed)
        assert first.dropped == second.dropped

    @staticmethod
    def with_neighbors(east: float) -> tuple:
        """
        Add commercial footprints of a second block group at a distance east.

        Args:
            east (float):
                Meters east of the fixture origin of the added footprints.

        Returns:
            tuple:
                The table with a row for the second block group, the
                mobile-home park points, the targets and the reached
                footprints.
        """
        rows, mh_parks, targets, reached = scene()
        rows = pd.concat([rows, table({BG_B: {'002': (5, 1), '010': (4, 1)}})])
        for i in range(4):
            targets.add_asset(
                f'f{i}', Asset(f'f{i}', *empty(east + 100, 40 * i, bg=BG_B))
            )
            features = in_bg(BG_B, **{OCC: 'COM1', STORIES: 2})
            reached.add_asset(
                f'g{i}', Asset(f'g{i}', *asset(east, 40 * i + 60, **features))
            )
        return rows, [*mh_parks, lonlat(east, 0)], targets, reached

    def test_locality(self) -> None:
        """Assets outside the block group and past the imputer's cap change nothing."""
        rows, mh_parks, targets, reached = scene()
        base = budget(rows, mh_parks).allocate(targets, reached, n_worlds=WORLDS)
        # The nearest block-group footprint, b1, lies 2,050 m west of x = 3,600.
        wider = budget(*self.with_neighbors(3600.0)[:2]).allocate(
            *self.with_neighbors(3600.0)[2:], n_worlds=WORLDS
        )
        inside = {
            a: f
            for a, f in features_of(wider.imputed).items()
            if not a.startswith('f')
        }
        assert inside == features_of(base.imputed)
        log = wider.log.set_index('asset_id')
        assert (
            log.loc['b1', 'probability']
            == base.log.set_index('asset_id').loc['b1', 'probability']
        )

    def test_bound_reaches_the_cap(self) -> None:
        """A donor outside the block group but within the cap can move a rank."""
        rows, mh_parks, targets, reached = scene()
        base = budget(rows, mh_parks).allocate(targets, reached).log
        near = (
            budget(*self.with_neighbors(200.0)[:2])
            .allocate(*self.with_neighbors(200.0)[2:])
            .log
        )
        before = base.set_index('asset_id')['probability']
        after = near.set_index('asset_id')['probability'].reindex(before.index)
        assert (after != before).any()


class TestDisjointOutputs:
    """The imputer and the budget fill different features."""

    def test_disjoint(self) -> None:
        """The imputer fills only what the budget left, together filling all."""
        rows, mh_parks, targets, reached = scene()
        donor_imputer = imputer()
        result = budget(rows, mh_parks, imputer=donor_imputer).allocate(
            targets, reached, n_worlds=2, relabel_sources=RELABEL
        )
        staged = AssetInventory()
        budget_features = features_of(result.imputed)
        for asset_id, x in targets.inventory.items():
            if asset_id in result.dropped:
                continue
            features = {**x.features, **budget_features.get(asset_id, {})}
            staged.add_asset(asset_id, Asset(asset_id, x.coordinates, features))
        staged.n_pw = 2
        imputed = donor_imputer.impute_stage(
            staged, reached, n_worlds=2, status=result.residential
        ).imputed
        targets_only = {
            a: f for a, f in budget_features.items() if a in targets.inventory
        }
        for asset_id, x in imputed.inventory.items():
            assert not set(x.features) & set(targets_only.get(asset_id, {}))
        assert (imputed.n_pw, result.imputed.n_pw) == (2, 2)
        filled = set(imputed.inventory['t7'].features) | set(budget_features['t7'])
        assert filled == {OCC, UNITS, STORIES}


class TestCapacity:
    """No dwelling holds more units than its gross floor area allows."""

    def test_over_capacity_recorded_and_warned(self) -> None:
        """A 50-plus residual too large for its footprints is reported."""
        rows = table({BG_A: {'009': (300, 20)}})
        # Two footprints with room for 64 units each, and two with room for 3.
        targets = inventory(
            {
                'a': asset(0, 0, 30.0, **in_bg(**{STORIES: 2})),
                'b': asset(100, 0, 30.0, **in_bg(**{STORIES: 2})),
                's1': empty(200, 0),
                's2': empty(240, 0),
            }
        )
        with pytest.warns(UserWarning, match='In 1 of the block groups'):
            result = budget(rows, []).allocate(targets, inventory({}), n_worlds=2)
        gaps = gap_kinds(result)
        # 300 units need 3 structures, of which 2 find a footprint for 50.
        assert gaps[(BG_A, '50+', GAP_CANDIDATE_SHORTFALL)][0] == 1
        assert gaps[(BG_A, '50+', GAP_UNITS_OVER_CAPACITY)] == (0, 300 - 2 * 64)
        features = features_of(result.imputed)
        assert set(features) == {'a', 'b'}
        assert {features[a][UNITS] for a in features} == {64}
        assert set(result.nonresidential) == {'s1', 's2'}

    def test_stories_multiply_capacity(self) -> None:
        """The plan area times the story count gives the gross floor area."""
        rows = table({BG_A: {'002': (1, 1)}})
        targets = inventory(
            {'tall': asset(0, 0, **in_bg(**{STORIES: 3})), 'low': empty(100, 0)}
        )
        log = budget(rows, []).allocate(targets, inventory({})).log
        log = log.set_index('asset_id')
        area = log.loc['tall', 'plan_area_sqft']
        assert log.loc['tall', 'capacity'] == math.floor(area * 3 / 300)
        assert log.loc['tall', 'stories'] == 3
        assert log.loc['low', 'capacity'] == math.floor(area / 300)
        assert math.isnan(log.loc['low', 'stories'])
        sources = log['gross_area_source']
        assert (sources['tall'], sources['low']) == ('stories', 'plan_area')

    def test_gross_area_first(self) -> None:
        """A usable GrossArea gives the gross floor area before the stories."""
        rows = table({BG_A: {'002': (1, 1)}})
        targets = inventory(
            {
                'g': asset(0, 0, **in_bg(**{GROSS_AREA_KEY: 5000.0, STORIES: 3})),
                'z': asset(100, 0, **in_bg(**{GROSS_AREA_KEY: 0.0, STORIES: 3})),
            }
        )
        log = budget(rows, []).allocate(targets, inventory({})).log
        log = log.set_index('asset_id')
        assert log.loc['g', 'capacity'] == 5000 // 300
        assert math.isnan(log.loc['g', 'stories'])
        sources = log['gross_area_source']
        assert (sources['g'], sources['z']) == ('gross_area', 'stories')
        area = log.loc['z', 'plan_area_sqft']
        assert log.loc['z', 'capacity'] == math.floor(area * 3 / 300)

    def test_reservation_by_capacity(self) -> None:
        """A large bin goes to a footprint that holds it, not the higher-ranked."""
        rows = table({BG_A: {'002': (4, 1), '008': (25, 5)}})
        reached = inventory(
            {
                **{f'r{i}': house(0, 40 * i) for i in range(4)},
                # reached commercial footprints
                **{
                    f'c{i}': asset(
                        1500, 40 * i, 20.0, **in_bg(**{OCC: 'COM1', STORIES: 1})
                    )
                    for i in range(5)
                },
            }
        )
        targets = inventory({'small': empty(60, 0), 'large': empty(1550, 100, 30)})
        result = budget(rows, []).allocate(targets, reached, n_worlds=WORLDS)
        log = result.log.set_index('asset_id')
        assert log.loc['small', 'probability'] > log.loc['large', 'probability']
        assert log.loc['large', 'outcome'] == OUTCOME_DWELLING
        assert features_of(result.imputed) == {'large': {OCC: 'RES3E', UNITS: 25}}

    def test_placed_units_same_every_world(self) -> None:
        """Where no dwelling holds a bin's share, every world places as much."""
        rows = table({BG_A: {'002': (1, 1), '006': (9, 2)}})
        # Room for 5 and for 6 units, so neither holds the 9.
        targets = inventory({'a': empty(0, 0, 12), 'b': empty(100, 0, 13)})
        with pytest.warns(UserWarning, match='up to 3 units'):
            result = budget(rows, []).allocate(targets, inventory({}), n_worlds=12)
        diagnostic = result.diagnostic.set_index(['bin', 'world'])
        placed = {diagnostic.loc[('5-9', w), 'generated_units'] for w in range(12)}
        assert placed == {6.0}
        assert gap_kinds(result)[(BG_A, '5-9', GAP_UNITS_OVER_CAPACITY)] == (0, 3)

    def test_share_goes_where_it_fits(self) -> None:
        """A bin goes to the dwelling that holds its share, so every unit is placed."""
        rows = table({BG_A: {'002': (1, 1), '006': (9, 2)}})
        # Room for 5 and for 9 units: 1 + 9 fit in every world.
        targets = inventory({'a': empty(0, 0, 12), 'b': empty(100, 0, 16)})
        result = budget(rows, []).allocate(targets, inventory({}), n_worlds=12)
        total = result.diagnostic.set_index('bin').loc['total', 'generated_units']
        assert list(total) == [10.0] * 12
        assert GAP_UNITS_OVER_CAPACITY not in set(result.gaps['kind'])
        assert features_of(result.imputed)['b'] == {OCC: 'RES3C', UNITS: 9}


class TestCapacityOrder:
    """Gross floor area weights the per-world order and gates mobile homes."""

    def test_fallback_takes_the_largest(self) -> None:
        """Where none holds the share, the bin goes to the largest capacity."""
        rows = table({BG_A: {'006': (9, 2)}})
        # Room for 6 on the larger plan, which ranks first, and for 7 on two
        # stories of the smaller one.
        targets = inventory(
            {
                'wide': empty(0, 0, 13),
                'tall': asset(100, 0, **in_bg(**{STORIES: 2})),
            }
        )
        result = budget(rows, []).allocate(targets, inventory({}), n_worlds=WORLDS)
        assert features_of(result.imputed) == {'tall': {OCC: 'RES3C', UNITS: 7}}
        assert gap_kinds(result)[(BG_A, '5-9', GAP_UNITS_OVER_CAPACITY)] == (0, 2)

    def test_fallback_places_what_fits(self) -> None:
        """A bin short of dwellings for its share keeps the larger fallback ones."""
        rows = table({BG_A: {'002': (1, 1), '006': (18, 2)}})
        # Room for 9, 8 and 5 units: the 5-to-9 bin's two structures need
        # 9 each, so it takes 9 and 8 and the one-unit bin the 5.
        targets = inventory(
            {'a': empty(0, 0, 16), 'b': empty(100, 0, 15), 'c': empty(200, 0, 12)}
        )
        result = budget(rows, []).allocate(targets, inventory({}), n_worlds=12)
        total = result.diagnostic.set_index('bin').loc['total', 'generated_units']
        assert list(total) == [18.0] * 12
        assert features_of(result.imputed)['c'] == {OCC: 'RES1', UNITS: 1}

    def test_order_weights_gross_area(self) -> None:
        """Of two equal plans, the taller footprint takes the larger bin more often."""
        # Both have room for the 7 units, the low one at 14 m sides.
        rows = table({BG_A: {'002': (1, 1), '006': (7, 2)}})
        targets = inventory(
            {
                'tall': asset(0, 0, 14.0, **in_bg(**{STORIES: 10})),
                'low': empty(100, 0, 14),
            }
        )
        result = budget(rows, []).allocate(targets, inventory({}), n_worlds=40)
        tall = features_of(result.imputed)['tall'][OCC]
        # The tall footprint goes first with chance 10 / 11 per world.
        assert sum(occupancy == 'RES3C' for occupancy in tall) >= 32

    def test_mobile_home_needs_capacity(self) -> None:
        """A target too small for one unit does not become a mobile home."""
        rows = table({BG_A: {'010': (2, 1)}})
        # A 5 m side gives 269 sq ft, under one unit's 300.
        targets = inventory({'tiny': empty(-490, 0, 5), 'b': empty(-470, 0)})
        result = budget(rows, [lonlat(*MH_PARK)]).allocate(targets, inventory({}))
        mobile = {
            a for a, f in features_of(result.imputed).items() if f.get(OCC) == 'RES2'
        }
        assert mobile == {'b'}
        assert gap_kinds(result)[
            (BG_A, MOBILE_HOME_LABEL, GAP_MOBILE_HOME_SHORTFALL)
        ] == (1, 1)


class TestArguments:
    """Unusable arguments raise."""

    def test_negative_threshold(self) -> None:
        """A negative threshold raises."""
        with pytest.raises(ValueError, match='negative'):
            budget(table({}), [], threshold_sqft=-1.0)

    def test_unit_area_not_positive(self) -> None:
        """A gross area per unit of zero raises."""
        with pytest.raises(ValueError, match='positive'):
            budget(table({}), [], min_unit_area_sqft=0.0)

    def test_inventories(self) -> None:
        """Several worlds, shared assets and zero worlds raise."""
        rows, mh_parks, targets, reached = scene()
        allocate = budget(rows, mh_parks).allocate
        with pytest.raises(ValueError, match='n_worlds'):
            allocate(targets, reached, n_worlds=0)
        shared = deepcopy(reached)
        shared.add_asset('t0', Asset('t0', *empty(60, 0)))
        with pytest.raises(ValueError, match='share'):
            allocate(targets, shared)
        several = deepcopy(targets)
        several.n_pw = 2
        with pytest.raises(ValueError, match='one possible world'):
            allocate(several, reached)

    def test_mh_park_points(self) -> None:
        """Mobile-home park points must be points or longitude and latitude pairs."""
        with pytest.raises(ValueError, match='pairs'):
            budget(table({}), [[1.0, 2.0, 3.0]])

    @pytest.mark.parametrize(
        'status',
        [
            {'status_values': ('RES1',)},
            {
                'status_key': 'Material',
                'required': ('OccupancyClass', 'NumberOfStories', 'Material'),
            },
        ],
    )
    def test_imputer_of_another_status(self, status: dict[str, Any]) -> None:
        """An imputer whose status is not residential occupancy raises."""
        with pytest.raises(ValueError, match='residential'):
            budget(table({}), [], imputer=imputer(**status))


class TestImporter:
    """The class is found where callers look for it."""

    def test_importer(self) -> None:
        """The Importer resolves the class."""
        assert Importer().get_class('ACSUnitBudget') is ACSUnitBudget
