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

"""Tests for the re-draw of implausible occupancy and story count pairs."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable
from unittest.mock import Mock

import pytest

import brails.compilers
from brails.compilers.hazus_earthquake.pair_redraw import (
    ACTION_LOG_IMPUTED,
    ACTION_LOG_LOCAL,
    ACTION_REDRAW_OCCUPANCY,
    ACTION_REDRAW_PAIR,
    ACTION_REPLACE_STORIES,
    ACTION_REPLACE_STORIES_REDRAW_OCCUPANCY,
    LOG_COLUMNS,
    OUTCOME_LOGGED,
    PLAN_COLUMNS,
    PURPOSE_PAIR_REDRAW,
    PairRedraw,
    PairRedrawResult,
)
from brails.constants import BLOCK_GROUP_KEY
from brails.imputers import ObservedDonorImputer
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    OUTCOME_BLOCK_GROUP,
    OUTCOME_DONOR,
    OUTCOME_EXHAUSTED,
    OUTCOME_NEAREST_ADMITTED,
    OUTCOME_NO_AREA,
    OUTCOME_NO_DONOR,
    DonorDraw,
)
from brails.inferers import StoriesFromHeightInferer
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils import Importer
from brails.utils.source_updates import WITHHELD_COLUMNS, apply_to_sources
from tests.compilers.source_fixtures import (
    PRECEDENCE,
    codes,
    merge_sources,
    merged,
    nsi_tall_industry,
    sources_for_one_asset,
)
from tests.imputers.donor_fixtures import BG_A, BG_B, SEED, square

MICROSOFT = 'Microsoft ML Buildings'
OSM = 'OpenStreetMap'

# Scales of the stories inferer, not the packaged calibration.
FIXTURE_SCALES = {
    'length_unit': 'm',
    'providers': {
        'overture': {
            'sources': {
                MICROSOFT: {'scale': 3.92, 'max_stories': 2},
                OSM: {'scale': 4.0, 'max_stories': None},
            }
        }
    },
}
OSM_SCALE_M = FIXTURE_SCALES['providers']['overture']['sources'][OSM]['scale']
STORIES_INFERER = StoriesFromHeightInferer(length_unit='m', scales=FIXTURE_SCALES)

# The provenance of an Overture footprint whose height is OpenStreetMap's.
OSM_HEIGHT = {'footprint_provider': 'overture', 'sources': f'dataset: {OSM}'}

MAX_ATTEMPTS = 5

# The donors the keyed re-draw picks for keyed_target(), computed with SEED
# and SECOND_SEED, and with SEED under the imputer's own labels.
GOLDEN_DONOR = 'd4'
SECOND_SEED = 1
SECOND_SEED_DONOR = 'd2'
STAGE_LABEL_DONOR = 'd1'

# Candidate donor values the stub offers, in order.
NON_RESIDENTIAL_CANDIDATES = (
    {'OccupancyClass': 'IND1', 'NumberOfStories': 9},
    {'OccupancyClass': 'COM1', 'NumberOfStories': 9},
    {'OccupancyClass': 'COM4', 'NumberOfStories': 2},
)
RESIDENTIAL_CANDIDATES = (
    {'OccupancyClass': 'RES3A', 'NumberOfStories': 5},
    {'OccupancyClass': 'RES1', 'NumberOfStories': 5},
    {'OccupancyClass': 'RES2', 'NumberOfStories': 1},
)

# Iteration the stub reports for an accepted pick, distinct from any
# iteration the re-draw passes.
STUB_ITERATION = 41


class StubDraw:
    """
    Stand in for ``ObservedDonorImputer.draw`` and record its calls.

    The stub returns the first of its candidates that ``admit`` passes, or
    ``exhausted``. A forced outcome replaces the pick, with no donor unless
    it is the block-group fallback, and forced values are returned unjudged.
    """

    def __init__(
        self,
        outcomes: dict[str, str] | None = None,
        values: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        Configure the stub.

        Args:
            outcomes (dict[str, str], optional):
                A forced outcome per asset id.
            values (dict[str, dict[str, Any]], optional):
                Forced donor values per asset id.
        """
        self.outcomes = outcomes or {}
        self.values = values or {}
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        pool: object,
        asset_id: str,
        features: dict[str, Any],
        redraw: tuple[str, ...] | None = None,
        *,
        world: int,
        iteration: int = 0,
        admit: Callable[[dict[str, Any]], bool] | None = None,
        exclude_donors: tuple = (),
        status: bool | None = None,
        protected: tuple[str, ...] = (),
        scan_nearest: bool = True,
        purpose_prefix: str | None = None,
    ) -> DonorDraw:
        """
        Draw for one asset as the imputer's signature has it.

        Args:
            pool (object):
                The donor pool, recorded only.
            asset_id (str):
                The id of the asset drawn for, which selects a forced
                outcome or forced values.
            features (dict[str, Any]):
                The asset's features, recorded as a copy.
            redraw (tuple[str, ...], optional):
                The features to copy from the donor.
            world (int):
                The world the draw runs in, recorded and returned.
            iteration (int, optional):
                The first iteration, recorded only.
            admit (Callable, optional):
                The test each candidate's values must pass.
            exclude_donors (tuple, optional):
                Donors the draw may not pick, recorded only.
            status (bool, optional):
                The residential status, which selects the stub's
                candidates.
            protected (tuple[str, ...], optional):
                Features that may not change, recorded only.
            scan_nearest (bool, optional):
                Whether the draw ends with the nearest-first scan, recorded
                only.
            purpose_prefix (str, optional):
                The prefix of the purpose labels, recorded only.

        Returns:
            DonorDraw:
                The draw: the donor id, its values, the attempts and the
                outcome.
        """
        self.calls.append(
            {
                'pool': pool,
                'asset_id': asset_id,
                'features': dict(features),
                'redraw': tuple(redraw),
                'world': world,
                'iteration': iteration,
                'admit': admit,
                'exclude_donors': exclude_donors,
                'status': status,
                'protected': tuple(protected),
                'scan_nearest': scan_nearest,
                'purpose_prefix': purpose_prefix,
            }
        )
        forced = self.outcomes.get(asset_id)
        candidates = RESIDENTIAL_CANDIDATES if status else NON_RESIDENTIAL_CANDIDATES
        # The pick as (donor id, values, attempts, outcome).
        pick = (None, {}, len(candidates), OUTCOME_EXHAUSTED)
        if asset_id in self.values:
            pick = ('d9', self.values[asset_id], 1, OUTCOME_DONOR)
        elif forced is not None and forced != OUTCOME_BLOCK_GROUP:
            pick = (None, {}, 1, forced)
        else:
            for attempt, candidate in enumerate(candidates, start=1):
                values = {key: candidate[key] for key in redraw}
                if admit(values):
                    pick = (f'd{attempt}', values, attempt, forced or OUTCOME_DONOR)
                    break
        donor_id, values, attempts, outcome = pick
        return DonorDraw(
            asset_id=asset_id,
            world=world,
            iteration=STUB_ITERATION,
            donor_id=donor_id,
            values=values,
            level=0 if donor_id else None,
            n_candidates=3,
            distance=1.0 if donor_id else math.nan,
            distance_m=20.0 if donor_id else math.nan,
            attempts=attempts,
            outcome=outcome,
        )


def new_imputer(**options: Any) -> ObservedDonorImputer:
    """
    Return an imputer on the fixture seed.

    Args:
        **options (Any):
            Constructor arguments that replace the defaults.

    Returns:
        ObservedDonorImputer:
            The imputer, whose public ``nearest_admitted`` raises if called.
    """
    imputer = ObservedDonorImputer(
        **{'max_attempts': MAX_ATTEMPTS, 'seed': SEED, **options}
    )
    imputer.nearest_admitted = Mock(
        side_effect=AssertionError(
            'the pair re-draw called nearest_admitted directly'
        )
    )
    return imputer


@pytest.fixture
def imputer() -> ObservedDonorImputer:
    """
    Return the default fixture imputer.

    Returns:
        ObservedDonorImputer:
            The imputer of :func:`new_imputer`.
    """
    return new_imputer()


def stubbed(
    imputer: ObservedDonorImputer,
    outcomes: dict[str, str] | None = None,
    values: dict[str, dict[str, Any]] | None = None,
) -> tuple[PairRedraw, StubDraw]:
    """
    Return a re-draw whose imputer draws through a stub.

    Args:
        imputer (ObservedDonorImputer):
            The imputer, whose ``draw`` is replaced.
        outcomes (dict[str, str], optional):
            Forced outcomes per asset id.
        values (dict[str, dict[str, Any]], optional):
            Forced donor values per asset id.

    Returns:
        tuple[PairRedraw, StubDraw]:
            The re-draw, on no donors, and the stub.
    """
    stub = StubDraw(outcomes, values)
    imputer.draw = stub
    return PairRedraw(imputer, AssetInventory(), STORIES_INFERER), stub


class TestActionTable:
    """The action of each cell of the action table."""

    @pytest.mark.parametrize(
        ('occupancy_code', 'stories_code', 'height', 'action'),
        [
            ('N', 'L', None, ACTION_REDRAW_OCCUPANCY),
            ('N', 'F', None, ACTION_REDRAW_OCCUPANCY),
            ('N', 'N', 40.0, ACTION_REDRAW_OCCUPANCY),
            ('N', 'R', 40.0, ACTION_REDRAW_OCCUPANCY),
            ('N', 'R', None, ACTION_REDRAW_OCCUPANCY),
            ('N', 'R', 12.0, ACTION_REDRAW_OCCUPANCY),
            ('N', 'N', None, ACTION_REDRAW_PAIR),
            ('N', 'N', 12.0, ACTION_REPLACE_STORIES),
            ('N', 'I', None, ACTION_LOG_IMPUTED),
            ('L', 'L', None, ACTION_LOG_LOCAL),
            ('L', 'N', None, ACTION_LOG_LOCAL),
            ('F', 'N', 40.0, ACTION_LOG_LOCAL),
            ('F', 'R', None, ACTION_LOG_LOCAL),
            ('L', 'I', None, ACTION_LOG_IMPUTED),
            ('I', 'L', None, ACTION_LOG_IMPUTED),
            ('I', 'N', None, ACTION_LOG_IMPUTED),
            ('I', 'I', None, ACTION_LOG_IMPUTED),
        ],
    )
    def test_cell(
        self,
        imputer: ObservedDonorImputer,
        occupancy_code: str,
        stories_code: str,
        height: float | None,
        action: str,
    ) -> None:
        """Each pair of source codes takes its cell's action."""
        features = nsi_tall_industry(
            SourceString=codes(occupancy_code, '-', stories_code)
        )
        if height is not None:
            features.update(height=height, **OSM_HEIGHT)
        pair_redraw, _ = stubbed(imputer)
        plan = pair_redraw.plan(merged({'a': features}))
        assert list(plan.columns) == list(PLAN_COLUMNS)
        assert plan.to_dict('records') == [
            {
                'asset_id': 'a',
                'occupancy': 'IND1',
                'stories': 10,
                'band': 'high_rise',
                'occupancy_code': occupancy_code,
                'stories_code': stories_code,
                'height_stories': (
                    None if height is None else round(height / OSM_SCALE_M)
                ),
                'corroborated': height == 40.0,
                'action': action,
            }
        ]

    @pytest.mark.parametrize(
        ('occupancy_code', 'stories_code'),
        [
            ('R', 'N'),
            ('D', 'N'),
            ('-', 'N'),
            ('N', 'D'),
            ('N', '-'),
            ('D', 'I'),
            ('I', 'D'),
            ('L', 'D'),
            ('F', '-'),
        ],
    )
    def test_codes_outside_the_table_raise(
        self, imputer: ObservedDonorImputer, occupancy_code: str, stories_code: str
    ) -> None:
        """An implausible pair whose codes the action table does not cover raises."""
        features = nsi_tall_industry(
            SourceString=codes(occupancy_code, '-', stories_code)
        )
        pair_redraw, _ = stubbed(imputer)
        with pytest.raises(ValueError, match='does not cover'):
            pair_redraw.plan(merged({'a': features}))


class TestCorroboration:
    """A story count is corroborated by a height in its band."""

    def test_microsoft_does_not_corroborate_above_its_maximum(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """A Microsoft height of five stories does not corroborate, OSM's does."""
        sources = FIXTURE_SCALES['providers']['overture']['sources']
        five_stories_m = 5 * sources[MICROSOFT]['scale']
        records = {
            source: {
                'OccupancyClass': 'AGR1',
                'NumberOfStories': 5,
                'SourceString': codes(),
                'height': five_stories_m,
                'footprint_provider': 'overture',
                'sources': f'dataset: {source}',
            }
            for source in (MICROSOFT, OSM)
        }
        pair_redraw, _ = stubbed(imputer)
        plan = pair_redraw.plan(merged(records)).set_index('asset_id')
        assert plan.loc[MICROSOFT, 'height_stories'] is None
        assert not plan.loc[MICROSOFT, 'corroborated']
        assert plan.loc[MICROSOFT, 'action'] == ACTION_REDRAW_PAIR
        assert plan.loc[OSM, 'height_stories'] == 5
        assert plan.loc[OSM, 'corroborated']
        assert plan.loc[OSM, 'action'] == ACTION_REDRAW_OCCUPANCY

    def test_a_count_in_the_same_band_corroborates(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """A height implying eight stories corroborates a stated ten."""
        features = nsi_tall_industry(height=8 * OSM_SCALE_M, **OSM_HEIGHT)
        pair_redraw, _ = stubbed(imputer)
        plan = pair_redraw.plan(merged({'a': features}))
        assert plan.loc[0, 'height_stories'] == 8
        assert plan.loc[0, 'corroborated']
        assert plan.loc[0, 'action'] == ACTION_REDRAW_OCCUPANCY

    def test_overture_sources_name_the_height_source(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """The height source is read from the sources entry of the height."""
        features = nsi_tall_industry(
            height=40.0,
            footprint_provider='overture',
            sources=(
                f'property: , dataset: {MICROSOFT}; '
                f'property: /properties/height, dataset: {OSM}'
            ),
        )
        pair_redraw, _ = stubbed(imputer)
        plan = pair_redraw.plan(merged({'a': features}))
        assert plan.loc[0, 'corroborated']
        assert plan.loc[0, 'action'] == ACTION_REDRAW_OCCUPANCY

    def test_no_height_is_uncorroborated(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """An asset without a height key is uncorroborated, with no KeyError."""
        features = nsi_tall_industry(**OSM_HEIGHT)
        assert 'height' not in features
        pair_redraw, _ = stubbed(imputer)
        plan = pair_redraw.plan(merged({'a': features}))
        assert plan.loc[0, 'height_stories'] is None
        assert plan.loc[0, 'action'] == ACTION_REDRAW_PAIR

    def test_corroborated_story_count_is_never_redrawn(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """A corroborated NSI story count stays, and only the class is drawn."""
        features = nsi_tall_industry(height=40.0, **OSM_HEIGHT)
        pair_redraw, stub = stubbed(imputer)
        result = pair_redraw.redraw(merged({'a': features}))
        assert stub.calls[0]['redraw'] == ('OccupancyClass',)
        assert result.updates == {'a': {'OccupancyClass': 'COM1'}}
        assert result.withheld['feature'].tolist() == ['OccupancyClass']


class TestPlan:
    """What the plan lists and what it refuses."""

    def test_plausible_and_unreadable_assets_are_not_listed(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """Plausible pairs and assets whose class or count is unreadable."""
        records = {
            'com': nsi_tall_industry(OccupancyClass='COM1'),
            'low': nsi_tall_industry(NumberOfStories=2),
            'no_class': nsi_tall_industry(OccupancyClass='NA'),
            'no_stories': nsi_tall_industry(NumberOfStories=None),
            'half': nsi_tall_industry(NumberOfStories=9.5),
            'tall': nsi_tall_industry(),
        }
        pair_redraw, stub = stubbed(imputer)
        assert pair_redraw.plan(merged(records))['asset_id'].tolist() == ['tall']
        result = pair_redraw.redraw(merged(records))
        assert list(result.updates) == ['tall']
        assert [call['asset_id'] for call in stub.calls] == ['tall']

    def test_a_numeric_string_count_is_read(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """A count of '10', which the structure type inferer reads, is planned."""
        pair_redraw, _ = stubbed(imputer)
        result = pair_redraw.redraw(
            merged({'a': nsi_tall_industry(NumberOfStories='10')})
        )
        assert result.log.loc[0, 'stories'] == 10
        assert result.log.loc[0, 'action'] == ACTION_REDRAW_PAIR
        assert result.withheld.loc[1, 'value'] == '10'

    def test_plan_is_pure(self, imputer: ObservedDonorImputer) -> None:
        """The plan changes nothing, calls no draw and returns the same frame twice."""
        inv = merged({'a': nsi_tall_industry(), 'b': nsi_tall_industry()})
        before = deepcopy({k: a.features for k, a in inv.inventory.items()})
        pair_redraw, stub = stubbed(imputer)
        first = pair_redraw.plan(inv)
        assert first.equals(pair_redraw.plan(inv))
        assert {k: a.features for k, a in inv.inventory.items()} == before
        assert stub.calls == []

    @pytest.mark.parametrize('string', [None, 'NN', 'NNNNNNNNN'])
    def test_source_string_must_cover_the_features(
        self, imputer: ObservedDonorImputer, string: str | None
    ) -> None:
        """A missing or wrong-length source string raises, on any asset."""
        records = {
            'tall': nsi_tall_industry(),
            'plain': {'OccupancyClass': 'COM1', 'SourceString': string},
        }
        pair_redraw, _ = stubbed(imputer)
        with pytest.raises(ValueError, match='source string'):
            pair_redraw.plan(merged(records))

    def test_several_worlds_raise(self, imputer: ObservedDonorImputer) -> None:
        """A plan over an inventory of two possible worlds raises."""
        inv = merged({'a': nsi_tall_industry()})
        inv.n_pw = 2
        pair_redraw, _ = stubbed(imputer)
        with pytest.raises(ValueError, match='possible worlds'):
            pair_redraw.plan(inv)

    def test_source_features_must_hold_the_pair(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """The source features must be a sequence naming the pair once each."""
        with pytest.raises(ValueError, match='lacks'):
            PairRedraw(
                imputer,
                AssetInventory(),
                STORIES_INFERER,
                source_features=('OccupancyClass',),
            )
        with pytest.raises(ValueError, match='repeats'):
            PairRedraw(
                imputer,
                AssetInventory(),
                STORIES_INFERER,
                source_features=(
                    'OccupancyClass',
                    'NumberOfStories',
                    'OccupancyClass',
                ),
            )
        with pytest.raises(TypeError):
            PairRedraw(
                imputer,
                AssetInventory(),
                STORIES_INFERER,
                source_features='OccupancyClass',
            )

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
            PairRedraw(new_imputer(**status), AssetInventory(), STORIES_INFERER)


class TestDrawCall:
    """What the re-draw passes to the imputer's draw."""

    def test_arguments(self, imputer: ObservedDonorImputer) -> None:
        """The draw receives the asset's arguments and the log keeps the draw's."""
        features = nsi_tall_industry(
            NumberOfUnits=0, SourceString=codes('N', 'L', 'F')
        )
        stub = StubDraw()
        imputer.draw = stub
        pair_redraw = PairRedraw(imputer, AssetInventory(), STORIES_INFERER, world=3)
        result = pair_redraw.redraw(merged({'a': features}))
        (call,) = stub.calls
        assert call['pool'].imputer is imputer
        assert call['asset_id'] == 'a'
        assert call['features'] == features
        assert call['world'] == 3
        assert call['redraw'] == ('OccupancyClass',)
        assert call['status'] is False
        assert call['protected'] == ('NumberOfUnits', 'NumberOfStories')
        assert call['iteration'] == 0
        assert call['scan_nearest'] is True
        assert call['purpose_prefix'] == PURPOSE_PAIR_REDRAW
        log = result.log.loc[0]
        assert log['world'] == 3
        assert log['attempts'] == 2
        assert log['iteration'] == STUB_ITERATION
        assert log['donor_id'] == 'd2'

    def test_the_pool_is_built_from_the_donors(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """The re-draw builds its pool through imputer.pool, once per call."""
        donors = AssetInventory()
        built = []
        stage_pool = imputer.pool

        def spy(*args: Any) -> Any:
            """
            Record the call and build the pool.

            Args:
                *args (Any):
                    Passed to the imputer's pool.

            Returns:
                Any:
                    The pool the imputer's ``pool`` built.
            """
            built.append((args, stage_pool(*args)))
            return built[-1][1]

        imputer.pool = spy
        stub = StubDraw()
        imputer.draw = stub
        inventory = merged({'a': nsi_tall_industry(), 'b': nsi_tall_industry()})
        PairRedraw(imputer, donors, STORIES_INFERER).redraw(inventory)
        (((first, second), pool),) = built
        assert (first, second) == (donors, inventory)
        assert [call['pool'] for call in stub.calls] == [pool, pool]

    def test_pair_redraw_names_both(self, imputer: ObservedDonorImputer) -> None:
        """An uncorroborated NSI count is drawn with the class."""
        pair_redraw, stub = stubbed(imputer)
        result = pair_redraw.redraw(merged({'a': nsi_tall_industry()}))
        assert stub.calls[0]['redraw'] == ('OccupancyClass', 'NumberOfStories')
        assert stub.calls[0]['protected'] == ()
        assert result.updates == {
            'a': {'OccupancyClass': 'COM1', 'NumberOfStories': 9}
        }
        assert list(result.withheld.columns) == list(WITHHELD_COLUMNS)
        assert result.withheld.to_dict('records') == [
            {
                'asset_id': 'a',
                'feature': 'OccupancyClass',
                'source': 'nsi',
                'value': 'IND1',
            },
            {
                'asset_id': 'a',
                'feature': 'NumberOfStories',
                'source': 'nsi',
                'value': 10,
            },
        ]

    def test_derived_story_count_is_kept(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """A derived count is the height's own, so only the class is drawn."""
        pair_redraw, stub = stubbed(imputer)
        result = pair_redraw.redraw(
            merged({'a': nsi_tall_industry(SourceString=codes('N', '-', 'R'))})
        )
        assert stub.calls[0]['redraw'] == ('OccupancyClass',)
        assert result.withheld['source'].tolist() == ['nsi']
        assert result.derived == {}

    def test_residential_status_is_the_assets(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """A residential asset is drawn with residential set, and logs its class."""
        features = {
            'OccupancyClass': 'RES2',
            'NumberOfStories': 5,
            'NumberOfUnits': 1,
            'YearBuilt': 1962,
            'SourceString': codes('N', 'N', 'L'),
        }
        pair_redraw, stub = stubbed(imputer)
        result = pair_redraw.redraw(merged({'a': features}))
        assert stub.calls[0]['status'] is True
        assert result.updates == {'a': {'OccupancyClass': 'RES1'}}
        assert result.log.loc[0, 'occupancy'] == 'RES2'

    def test_one_draw_and_no_direct_scan_call(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """An exhausted re-draw makes one draw call and no scan call of its own."""
        pair_redraw, stub = stubbed(imputer, {'a': OUTCOME_EXHAUSTED})
        pair_redraw.redraw(merged({'a': nsi_tall_industry()}))
        assert len(stub.calls) == 1

    def test_a_donor_without_the_drawn_values_raises(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """A draw returning a donor without a re-drawn feature raises."""
        pair_redraw, _ = stubbed(imputer, values={'a': {'OccupancyClass': 'COM1'}})
        with pytest.raises(RuntimeError, match='without'):
            pair_redraw.redraw(merged({'a': nsi_tall_industry()}))


# An eight-story NSI AGR1 whose height gives five stories. AGR1 is
# implausible above low-rise, so the pair stays implausible at the height
# count.
EIGHT_STORY_AGR1 = {
    'OccupancyClass': 'AGR1',
    'NumberOfStories': 8,
    'SourceString': codes(),
    'height': 5 * OSM_SCALE_M,
    **OSM_HEIGHT,
}


class TestHeightCount:
    """An uncorroborated NSI count is replaced by the height's count."""

    def test_replace_without_a_draw(self, imputer: ObservedDonorImputer) -> None:
        """With a plausible pair at the height count, only the count changes."""
        features = nsi_tall_industry(height=12.0, **OSM_HEIGHT)
        pair_redraw, stub = stubbed(imputer)
        result = pair_redraw.redraw(merged({'a': features}))
        assert stub.calls == []
        assert result.updates == {}
        assert result.derived == {'a': {'NumberOfStories': 3}}
        assert result.withheld.to_dict('records') == [
            {
                'asset_id': 'a',
                'feature': 'NumberOfStories',
                'source': 'nsi',
                'value': 10,
            }
        ]
        log = result.log.loc[0]
        assert (log['action'], log['outcome']) == (
            ACTION_REPLACE_STORIES,
            OUTCOME_LOGGED,
        )

    def test_replace_and_redraw_the_class(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """With the pair still implausible, the class is drawn against the height count."""
        pair_redraw, stub = stubbed(imputer)
        result = pair_redraw.redraw(merged({'a': EIGHT_STORY_AGR1}))
        (call,) = stub.calls
        assert call['redraw'] == ('OccupancyClass',)
        assert not call['admit']({'OccupancyClass': 'AGR1'})
        assert call['admit']({'OccupancyClass': 'IND1'})
        assert result.log.loc[0, 'action'] == ACTION_REPLACE_STORIES_REDRAW_OCCUPANCY
        assert result.updates == {'a': {'OccupancyClass': 'IND1'}}
        assert result.derived == {'a': {'NumberOfStories': 5}}
        assert result.withheld['feature'].tolist() == [
            'OccupancyClass',
            'NumberOfStories',
        ]

    def test_an_exhausted_class_draw_keeps_the_replaced_count(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """Without a donor the class stays and the count is replaced."""
        pair_redraw, _ = stubbed(imputer, {'a': OUTCOME_EXHAUSTED})
        result = pair_redraw.redraw(merged({'a': EIGHT_STORY_AGR1}))
        assert result.updates == {}
        assert result.derived == {'a': {'NumberOfStories': 5}}
        assert result.withheld['feature'].tolist() == ['NumberOfStories']
        assert result.log.loc[0, 'outcome'] == OUTCOME_EXHAUSTED

    def test_merge_codes_the_height_count_as_derived(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """The height count, written into the derived source and merged, codes R."""
        sources = sources_for_one_asset()
        sources['footprint'].inventory['a'].features.update(OSM_HEIGHT)
        pair_redraw, _ = stubbed(imputer)
        result = pair_redraw.redraw(merge_sources(sources))
        written = apply_to_sources(
            sources, result.updates, result.withheld, precedence=PRECEDENCE
        )
        written = apply_to_sources(
            written,
            result.derived,
            result.withheld.iloc[0:0],
            imputed_label='derived',
            precedence=PRECEDENCE,
        )
        merged_asset = merge_sources(written).inventory['a'].features
        assert merged_asset['NumberOfStories'] == 3
        assert merged_asset['SourceString'] == 'N-RL----'


class TestAdmit:
    """The test a candidate donor's values must pass."""

    def test_class_against_the_kept_count(
        self, imputer: ObservedDonorImputer
    ) -> None:
        """A class implausible at the kept count, or residential, is refused."""
        pair_redraw, stub = stubbed(imputer)
        features = nsi_tall_industry(SourceString=codes('N', '-', 'L'))
        pair_redraw.redraw(merged({'a': features}))
        admit = stub.calls[0]['admit']
        assert not admit({'OccupancyClass': 'IND2'})
        assert not admit({'OccupancyClass': 'RES3F'})
        assert not admit({})
        assert not admit({'OccupancyClass': None})
        assert admit({'OccupancyClass': 'COM1'})

    def test_pair(self, imputer: ObservedDonorImputer) -> None:
        """A re-drawn pair must be plausible as a pair."""
        pair_redraw, stub = stubbed(imputer)
        pair_redraw.redraw(merged({'a': nsi_tall_industry()}))
        admit = stub.calls[0]['admit']
        assert admit({'OccupancyClass': 'IND1', 'NumberOfStories': 3})
        assert admit({'OccupancyClass': 'IND1', 'NumberOfStories': '3'})
        assert not admit({'OccupancyClass': 'IND1', 'NumberOfStories': 8})
        assert not admit({'OccupancyClass': 'IND1', 'NumberOfStories': 0})
        assert not admit({'OccupancyClass': 'IND1'})

    def test_units_class(self, imputer: ObservedDonorImputer) -> None:
        """A residential class must agree with the asset's unit count."""
        features = {
            'OccupancyClass': 'RES2',
            'NumberOfStories': 5,
            'NumberOfUnits': 6,
            'SourceString': codes('N', 'N', 'N'),
        }
        pair_redraw, stub = stubbed(imputer)
        pair_redraw.redraw(merged({'a': features}))
        admit = stub.calls[0]['admit']
        assert admit({'OccupancyClass': 'RES3C', 'NumberOfStories': 5})
        assert not admit({'OccupancyClass': 'RES3D', 'NumberOfStories': 5})
        assert not admit({'OccupancyClass': 'RES1', 'NumberOfStories': 5})
        assert not admit({'OccupancyClass': 'COM1', 'NumberOfStories': 5})
        assert admit({'OccupancyClass': 'RES2', 'NumberOfStories': 2})
        assert not admit({'OccupancyClass': 'RES2', 'NumberOfStories': 4})

    @pytest.mark.parametrize('units', ['NA', 0, None])
    def test_no_positive_unit_count(
        self, imputer: ObservedDonorImputer, units: object
    ) -> None:
        """Without a positive unit count every residential class agrees."""
        features = {
            'OccupancyClass': 'RES2',
            'NumberOfStories': 5,
            'NumberOfUnits': units,
            'SourceString': codes('N', '-', 'L'),
        }
        pair_redraw, stub = stubbed(imputer)
        pair_redraw.redraw(merged({'a': features}))
        admit = stub.calls[0]['admit']
        assert admit({'OccupancyClass': 'RES3F'})
        assert admit({'OccupancyClass': 'RES1'})


class TestOutcomes:
    """What each draw outcome leaves and logs."""

    @pytest.mark.parametrize(
        'outcome', [OUTCOME_NO_DONOR, OUTCOME_NO_AREA, OUTCOME_EXHAUSTED]
    )
    def test_no_donor_adds_no_update(
        self, imputer: ObservedDonorImputer, outcome: str
    ) -> None:
        """An outcome without a donor adds no update and is logged."""
        pair_redraw, _ = stubbed(imputer, {'a': outcome})
        result = pair_redraw.redraw(merged({'a': nsi_tall_industry()}))
        assert result.updates == {}
        assert result.withheld.empty
        assert result.log.loc[0, 'outcome'] == outcome
        assert result.log.loc[0, 'donor_id'] is None

    def test_fallback_is_logged(self, imputer: ObservedDonorImputer) -> None:
        """A block-group fallback donor is used and logged as such."""
        pair_redraw, _ = stubbed(imputer, {'a': OUTCOME_BLOCK_GROUP})
        result = pair_redraw.redraw(merged({'a': nsi_tall_industry()}))
        assert result.log.loc[0, 'outcome'] == OUTCOME_BLOCK_GROUP
        assert 'a' in result.updates

    @pytest.mark.parametrize(
        ('source_string', 'action'),
        [
            (codes('L', '-', 'N'), ACTION_LOG_LOCAL),
            (codes('F', '-', 'L'), ACTION_LOG_LOCAL),
            (codes('I', '-', 'N'), ACTION_LOG_IMPUTED),
            (codes('N', '-', 'I'), ACTION_LOG_IMPUTED),
        ],
    )
    def test_logged_assets_are_unchanged(
        self, imputer: ObservedDonorImputer, source_string: str, action: str
    ) -> None:
        """Local, footprint and imputed pairs are logged and left."""
        pair_redraw, stub = stubbed(imputer)
        result = pair_redraw.redraw(
            merged({'a': nsi_tall_industry(SourceString=source_string)})
        )
        assert stub.calls == []
        assert result.updates == {}
        assert result.withheld.empty
        assert result.log.loc[0, 'action'] == action
        assert result.log.loc[0, 'outcome'] == OUTCOME_LOGGED
        assert result.log.loc[0, 'attempts'] == 0
        assert result.log.loc[0, 'iteration'] is None

    def test_log_columns(self, imputer: ObservedDonorImputer) -> None:
        """The log carries the plan's columns and the draw's."""
        pair_redraw, _ = stubbed(imputer)
        result = pair_redraw.redraw(merged({'a': nsi_tall_industry()}))
        assert isinstance(result, PairRedrawResult)
        assert list(result.log.columns) == list(LOG_COLUMNS)


class TestOrder:
    """The result does not depend on the order of the inventory."""

    def test_reordering(self, imputer: ObservedDonorImputer) -> None:
        """Two insertion orders give the same plan and result."""
        records = {
            'c': nsi_tall_industry(),
            'a': nsi_tall_industry(height=40.0, **OSM_HEIGHT),
            'b': nsi_tall_industry(SourceString=codes('L', '-', 'N')),
            'd': nsi_tall_industry(OccupancyClass='COM1'),
        }
        pair_redraw, _ = stubbed(imputer)
        first = pair_redraw.redraw(merged(records, ['c', 'a', 'd', 'b']))
        second = pair_redraw.redraw(merged(records, ['b', 'd', 'a', 'c']))
        assert list(first.updates.items()) == list(second.updates.items())
        assert list(first.updates) == ['a', 'c']
        assert first.withheld.equals(second.withheld)
        assert first.log.equals(second.log)
        assert first.log['asset_id'].tolist() == ['a', 'b', 'c']


def keyed_target() -> tuple[AssetInventory, AssetInventory]:
    """
    Return a ten-story NSI IND1 target and five plausible donors east of it.

    Returns:
        tuple[AssetInventory, AssetInventory]:
            The merged target inventory and the donor inventory.
    """
    donors = AssetInventory()
    specs = (
        ('d1', 15.0, 'COM1'),
        ('d2', 30.0, 'COM2'),
        ('d3', 45.0, 'COM4'),
        ('d4', 60.0, 'COM5'),
        ('d5', 75.0, 'GOV1'),
    )
    for asset_id, east, occupancy in specs:
        features = {'OccupancyClass': occupancy, 'NumberOfStories': 10}
        donors.add_asset(asset_id, Asset(asset_id, square(east, 0.0), features))
    target = merged({'t': nsi_tall_industry(SourceString=codes('N', '-', 'L'))})
    return target, donors


# Ten-story donors as (id, meters east, class, block group) for the
# locality tests, whose target stands at the origin in BG_A, and the seed
# and pick of those tests.
OUTSIDE_THE_CAP = (
    ('c1', 1900.0, 'COM1', BG_A),
    ('c2', 1950.0, 'COM2', BG_A),
    ('out', 2050.0, 'COM4', BG_B),
)
NO_ADMISSIBLE_DONOR = (
    ('i1', 1900.0, 'IND1', BG_A),
    ('other', 2100.0, 'COM2', BG_B),
)
LOCALITY_SEED = 2
LOCALITY_DONOR = 'c2'


def bounded_pick(
    donor_specs: tuple[tuple[str, float, str, str], ...],
    *,
    cap_m: float = 2000.0,
) -> tuple[Any, str]:
    """
    Re-draw the class of a ten-story NSI IND1 target in BG_A.

    Args:
        donor_specs (tuple[tuple[str, float, str, str], ...]):
            ``(id, meters east, class, block group)`` of ten-story donors.
        cap_m (float, optional):
            The imputer's donor cap. Defaults to 2000.

    Returns:
        tuple[Any, str]:
            The donor id and the outcome.
    """
    donors = AssetInventory()
    for asset_id, east, occupancy, block_group in donor_specs:
        features = {
            'OccupancyClass': occupancy,
            'NumberOfStories': 10,
            BLOCK_GROUP_KEY: block_group,
        }
        donors.add_asset(asset_id, Asset(asset_id, square(east, 0.0), features))
    features = nsi_tall_industry(SourceString=codes('N', '-', 'L'))
    features[BLOCK_GROUP_KEY] = BG_A
    target = merged({'t': features})
    imputer = new_imputer(cap_m=cap_m, seed=LOCALITY_SEED)
    log = PairRedraw(imputer, donors, STORIES_INFERER).redraw(target).log
    return log.loc[0, 'donor_id'], log.loc[0, 'outcome']


class TestWithTheImputer:
    """The re-draw against the imputer's own pool and draw."""

    def test_nearest_scan_finds_the_admissible_donor(self) -> None:
        """With one attempt at the nearest donor spent, the scan finds one."""
        imputer = new_imputer(k=1, max_attempts=1)
        donors = AssetInventory()
        for asset_id, east, features in (
            ('d1', 15.0, {'OccupancyClass': 'IND1', 'NumberOfStories': 10}),
            ('d2', 30.0, {'OccupancyClass': 'COM1', 'NumberOfStories': 10}),
            ('d3', 20.0, {'OccupancyClass': 'RES1', 'NumberOfStories': 1}),
        ):
            donors.add_asset(asset_id, Asset(asset_id, square(east, 0.0), features))
        target = merged({'t': nsi_tall_industry(SourceString=codes('N', '-', 'L'))})
        result = PairRedraw(imputer, donors, STORIES_INFERER).redraw(target)
        assert result.updates == {'t': {'OccupancyClass': 'COM1'}}
        log = result.log.loc[0]
        assert log['outcome'] == OUTCOME_NEAREST_ADMITTED
        assert log['donor_id'] == 'd2'

    @pytest.mark.parametrize(
        ('seed', 'donor'),
        [(SEED, GOLDEN_DONOR), (SECOND_SEED, SECOND_SEED_DONOR)],
        ids=['fixture_seed', 'second_seed'],
    )
    def test_golden_donor(self, seed: int, donor: str) -> None:
        """The keyed re-draw picks a pinned donor under each seed."""
        target, donors = keyed_target()
        imputer = new_imputer(seed=seed)
        log = PairRedraw(imputer, donors, STORIES_INFERER).redraw(target).log
        assert log.loc[0, 'donor_id'] == donor

    def test_the_redraw_draws_under_its_own_labels(self) -> None:
        """Without the re-draw's prefix the same pick takes another donor."""
        imputer = new_imputer()
        stage_draw = imputer.draw
        # Draw under the imputer's own labels.
        imputer.draw = lambda *args, **kwargs: stage_draw(
            *args, **{**kwargs, 'purpose_prefix': None}
        )
        target, donors = keyed_target()
        log = PairRedraw(imputer, donors, STORIES_INFERER).redraw(target).log
        assert log.loc[0, 'donor_id'] == STAGE_LABEL_DONOR

    def test_locality_bound(self) -> None:
        """A donor beyond the cap and outside the block group changes nothing."""
        # Two admissible donors lie within the cap, fewer than k, so a wider
        # cap lets the donor beyond it into the candidates and changes the pick.
        assert bounded_pick(OUTSIDE_THE_CAP)[0] == LOCALITY_DONOR
        assert bounded_pick(OUTSIDE_THE_CAP[:2])[0] == LOCALITY_DONOR
        changed = (*OUTSIDE_THE_CAP[:2], ('out', 2050.0, 'IND1', BG_B))
        assert bounded_pick(changed)[0] == LOCALITY_DONOR
        same_group = (*OUTSIDE_THE_CAP[:2], ('out', 2050.0, 'COM4', BG_A))
        assert bounded_pick(same_group)[0] == LOCALITY_DONOR
        assert bounded_pick(OUTSIDE_THE_CAP, cap_m=2100.0)[0] == 'out'

    def test_block_group_fallback(self) -> None:
        """Once no donor within the cap passes, a donor of the target's block group is drawn."""
        # A donor beyond the cap in another block group is never drawn.
        within_the_group = (*NO_ADMISSIBLE_DONOR, ('bg', 2500.0, 'COM1', BG_A))
        assert bounded_pick(within_the_group) == ('bg', OUTCOME_BLOCK_GROUP)
        assert bounded_pick(NO_ADMISSIBLE_DONOR) == (None, OUTCOME_EXHAUSTED)


def test_merge_codes_the_redrawn_values_as_imputed(
    imputer: ObservedDonorImputer,
) -> None:
    """The result, written into the sources and merged, codes I."""
    sources = sources_for_one_asset()
    pair_redraw, _ = stubbed(imputer)
    result = pair_redraw.redraw(merge_sources(sources))
    written = apply_to_sources(
        sources, result.updates, result.withheld, precedence=PRECEDENCE
    )
    assert merge_sources(written).inventory['a'].features['SourceString'] == (
        'I-IL----'
    )


def test_discovered_and_exported() -> None:
    """The Importer finds the class, and the package exports it."""
    assert Importer().get_class('PairRedraw') is PairRedraw
    assert 'PairRedraw' in brails.compilers.__all__
    assert brails.compilers.PairRedraw is PairRedraw
