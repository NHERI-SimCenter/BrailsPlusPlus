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

"""Tests for the composed screen and the resolution loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import brails.compilers
from brails.compilers.compilation_report import (
    EXCLUSION_CATEGORIES,
    CompilationReport,
)
from brails.compilers.hazus_earthquake.resolution_loop import (
    SCREEN_ASSET_ID,
    STAGE_DESIGN_LEVEL,
    STAGE_RESOLUTION,
    STAGE_STRUCTURE_TYPE,
    ComposedScreen,
    ResolutionLoop,
    ResolutionResult,
    ScreenVerdict,
    _category,
)
from brails.constants import (
    BLOCK_GROUP_KEY,
    BUILDING_TYPE_KEY,
    DESIGN_LEVEL_KEY,
    OCCUPANCY_KEY,
    STORIES_KEY,
    YEAR_BUILT_KEY,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_design_level_inferer import (
    HazusEQDesignLevelInferer,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_structure_type_inferer import (
    HazusEQStructureTypeInferer,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INFEASIBLE,
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
    diagnosis_frame,
)
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils import Importer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd

# The site every inferer of these tests is set up for.
STATE = 'California'
SEISMIC_ZONE = '4'

# Fixed so that the structure type inferer is reproducible.
TYPE_SEED = 12345

# A low-rise store the tables can take.
STORE = {
    OCCUPANCY_KEY: 'COM1',
    STORIES_KEY: 2,
    YEAR_BUILT_KEY: 1960,
}

# The same store with no construction year.
STORE_WITHOUT_YEAR = {OCCUPANCY_KEY: 'COM1', STORIES_KEY: 2}

# A store whose year no table can read.
UNREADABLE_YEAR_STORE = {**STORE, YEAR_BUILT_KEY: 'unknown'}

# A wood high-rise. Its cell, the combination of occupancy class, story band
# and era that selects a structure type distribution, is printed in the
# tables and holds no wood type.
WOOD_TOWER = {
    OCCUPANCY_KEY: 'RES3',
    STORIES_KEY: 9,
    YEAR_BUILT_KEY: 1960,
    BUILDING_TYPE_KEY: 'W',
}

# A wood store at eight stories. The tables print no distribution for its
# cell, and the distribution the completed tables fill it with holds no wood
# type.
TALL_WOOD_STORE = {**STORE, STORIES_KEY: 8, BUILDING_TYPE_KEY: 'W'}

# The world the loop's tests resolve, other than the first so that a
# record of the wrong world shows.
WORLD = 3

# The Census block group an asset states.
BLOCK_GROUP = '060750101001'

# A masonry walk-up whose refusal names its occupancy, story count, year
# and building type. Without the building type the tables can take it.
MASONRY_WALKUP = {
    OCCUPANCY_KEY: 'COM8',
    STORIES_KEY: 5,
    YEAR_BUILT_KEY: 1945,
    BUILDING_TYPE_KEY: 'M',
    BLOCK_GROUP_KEY: BLOCK_GROUP,
}

# The columns of a ledger row the loop's tests compare.
LEDGER_COLUMNS = [
    'asset_id',
    'world',
    'stage',
    'action',
    'feature',
    'old',
    'new',
    'source',
    'block_group',
]


def _screen(*, complete_tables: bool = True, **components: object) -> ComposedScreen:
    """
    Build a screen over the real earthquake inferers.

    Args:
        complete_tables (bool):
            Whether the structure type inferer draws from the completed
            West Coast tables.
        **components (object):
            Stages to put in place of the real inferers, by constructor
            argument name.

    Returns:
        ComposedScreen:
            The screen, with ``components`` in place of the real inferers
            they name.
    """
    settings = {
        'design_level_inferer': HazusEQDesignLevelInferer(
            state=STATE, seismic_zone=SEISMIC_ZONE
        ),
        'structure_type_inferer': HazusEQStructureTypeInferer(
            STATE, seed=TYPE_SEED, world=WORLD, complete_tables=complete_tables
        ),
    }
    settings.update(components)
    return ComposedScreen(**settings)


class _Stage:
    """
    A stage whose diagnosis a test writes, recording what it was asked.

    It reads the features the real inferers read, under their names.

    Attributes:
        year_built_key (str):
            Feature name of the construction year.
        design_level_key (str):
            Feature name of the design level.
        verdict (tuple):
            The diagnosis, detail and resolvable features every asset
            gets.
        asked (list[dict]):
            For each call of :meth:`check`, the features of the inventory
            it was given, by asset id.
    """

    year_built_key = YEAR_BUILT_KEY
    design_level_key = DESIGN_LEVEL_KEY

    def __init__(self, verdict: tuple) -> None:
        """
        Set up the stage with the verdict it returns.

        Args:
            verdict (tuple):
                The diagnosis, the detail, and the resolvable features.
        """
        self.verdict = verdict
        self.asked = []

    def check(self, input_inventory: AssetInventory) -> pd.DataFrame:
        """
        Give every asset of an inventory the same verdict.

        Args:
            input_inventory (AssetInventory):
                The inventory to diagnose.

        Returns:
            pd.DataFrame:
                One row per asset id.
        """
        asset_ids = list(input_inventory.get_asset_ids())
        self.asked.append(
            {
                asset_id: input_inventory.get_asset_features(asset_id)[1]
                for asset_id in asset_ids
            }
        )
        return diagnosis_frame(dict.fromkeys(asset_ids, self.verdict))


def test_an_asset_that_passes_every_stage_names_no_candidates() -> None:
    """An asset the whole chain can take is diagnosed OK and named none."""
    verdict = _screen().screen(STORE)

    assert verdict == ScreenVerdict(DIAGNOSIS_OK, '', '', ())


@pytest.mark.parametrize(
    'features', [STORE_WITHOUT_YEAR, UNREADABLE_YEAR_STORE], ids=('no', 'bad')
)
def test_a_year_the_table_cannot_read_is_refused_first(features: dict) -> None:
    """A missing or unusable year is refused where the level is assigned."""
    verdict = _screen().screen(features)

    assert verdict.diagnosis == DIAGNOSIS_INVALID
    assert verdict.stage == STAGE_DESIGN_LEVEL
    assert verdict.candidates == (YEAR_BUILT_KEY,)


def test_an_infeasible_cell_is_refused_at_the_structure_type_stage() -> None:
    """An asset with a usable year and an empty cell names its features."""
    verdict = _screen().screen(WOOD_TOWER)

    assert verdict.diagnosis == DIAGNOSIS_INFEASIBLE
    assert verdict.stage == STAGE_STRUCTURE_TYPE
    assert verdict.candidates == (
        OCCUPANCY_KEY,
        STORIES_KEY,
        BUILDING_TYPE_KEY,
    )


def test_the_screen_reads_the_completed_tables() -> None:
    """A cell only the completed tables fill passes with them, not without."""
    features = {**STORE, STORIES_KEY: 8}

    assert _screen().screen(features).diagnosis == DIAGNOSIS_OK
    assert (
        _screen(complete_tables=False).screen(features).diagnosis
        == DIAGNOSIS_INFEASIBLE
    )


def test_a_named_design_level_becomes_the_year() -> None:
    """A design level is changed through the year it is derived from."""
    structure_type = _Stage(
        (
            DIAGNOSIS_INFEASIBLE,
            'nothing to draw',
            (DESIGN_LEVEL_KEY, YEAR_BUILT_KEY, STORIES_KEY),
        )
    )

    verdict = _screen(structure_type_inferer=structure_type).screen(STORE)

    assert verdict.candidates == (YEAR_BUILT_KEY, STORIES_KEY)


def test_the_chain_stops_at_the_first_stage_that_refuses() -> None:
    """The second stage is never asked about an asset the first refused."""
    structure_type = _Stage((DIAGNOSIS_OK, '', ()))
    screen = _screen(structure_type_inferer=structure_type)

    verdict = screen.screen(UNREADABLE_YEAR_STORE)

    assert verdict.stage == STAGE_DESIGN_LEVEL
    assert not any(structure_type.asked)


@pytest.mark.parametrize(
    ('features', 'level'),
    [
        (STORE, 'Moderate-Code'),
        ({**STORE, DESIGN_LEVEL_KEY: 'Pre-Code'}, 'Pre-Code'),
    ],
    ids=('derived', 'given'),
)
def test_the_screen_reads_a_copy_under_its_own_id(
    features: dict, level: str
) -> None:
    """The second stage reads the given level, or a derived one the caller never sees."""
    structure_type = _Stage((DIAGNOSIS_OK, '', ()))
    given = dict(features)

    _screen(structure_type_inferer=structure_type).screen(given)

    assert given == features
    assert structure_type.asked == [
        {SCREEN_ASSET_ID: {**features, DESIGN_LEVEL_KEY: level}}
    ]


def test_an_inventory_screened_at_once_gets_each_asset_its_own_verdict(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """One check per stage gives every asset the verdict it gets alone."""
    features = {
        'passes': STORE,
        'given_level': {**STORE, DESIGN_LEVEL_KEY: 'Pre-Code'},
        'no_year': STORE_WITHOUT_YEAR,
        'wood_tower': WOOD_TOWER,
        'tall_wood_store': TALL_WOOD_STORE,
    }
    inventory = build_inventory(features, vector_features=[])
    screen = _screen()

    verdicts = screen.screen_inventory(inventory)

    assert verdicts == {
        asset_id: screen.screen(asset) for asset_id, asset in features.items()
    }
    assert [verdict.diagnosis for verdict in verdicts.values()] == [
        DIAGNOSIS_OK,
        DIAGNOSIS_OK,
        DIAGNOSIS_INVALID,
        DIAGNOSIS_INFEASIBLE,
        DIAGNOSIS_INFEASIBLE,
    ]
    assert inventory.get_asset_features('passes')[1] == STORE


@pytest.mark.parametrize('key', ['year_built_key', 'design_level_key'])
def test_inferers_that_name_a_driver_differently_are_refused(key: str) -> None:
    """The second stage must read the year and the level the first one reads."""
    structure_type = HazusEQStructureTypeInferer(
        STATE, seed=TYPE_SEED, complete_tables=True, **{key: 'Other'}
    )

    with pytest.raises(ValueError, match=key):
        _screen(structure_type_inferer=structure_type)


@pytest.mark.parametrize(
    ('features', 'complete_tables', 'acted_on', 'expected'),
    [
        (TALL_WOOD_STORE, True, False, 'conflict'),
        (TALL_WOOD_STORE, False, False, 'empty_cell'),
        (WOOD_TOWER, True, False, 'conflict'),
        (TALL_WOOD_STORE, False, True, 'conflict'),
        (UNREADABLE_YEAR_STORE, True, True, 'missing_driver'),
        ({**STORE, YEAR_BUILT_KEY: [1960, 1970]}, True, False, 'unsupported'),
    ],
    ids=(
        'filled_row_emptied',
        'no_row',
        'printed_row_emptied',
        'acted_on',
        'unreadable',
        'per_world_list',
    ),
)
def test_the_exclusion_category(
    features: dict,
    complete_tables: bool,  # noqa: FBT001
    acted_on: bool,  # noqa: FBT001
    expected: str,
) -> None:
    """Each refusal is filed in the category its cause gives."""
    screen = _screen(complete_tables=complete_tables)
    verdict = screen.screen(features)

    category = _category(screen, verdict, features, acted_on=acted_on)

    assert category == expected
    assert category in EXCLUSION_CATEGORIES


def test_an_infeasible_refusal_at_another_stage_is_undetermined() -> None:
    """Only the structure type stage selects a cell to look up."""
    verdict = ScreenVerdict(DIAGNOSIS_INFEASIBLE, '', STAGE_DESIGN_LEVEL)

    category = _category(_screen(), verdict, UNREADABLE_YEAR_STORE, acted_on=False)

    assert category == 'undetermined'


class _ScriptedScreen(ComposedScreen):
    """
    A screen whose verdicts a test writes from the features it is given.

    Attributes:
        rule (Callable[[dict], ScreenVerdict]):
            Takes the features of one asset and returns the verdict for
            them.
    """

    def __init__(self, rule: Callable[[dict], ScreenVerdict]) -> None:
        """
        Set up the screen over the real inferers, with the rule it applies.

        Args:
            rule (Callable[[dict], ScreenVerdict]):
                Takes the features and returns the verdict.
        """
        real = _screen()
        super().__init__(real.design_level_inferer, real.structure_type_inferer)
        self.rule = rule

    def screen_inventory(self, inventory: AssetInventory) -> dict:
        """
        Apply the test's rule to each asset of an inventory.

        Args:
            inventory (AssetInventory):
                The assets to screen.

        Returns:
            dict:
                Asset id to whatever the rule returns for its features.
        """
        return {
            asset_id: self.rule(dict(inventory.get_asset_features(asset_id)[1]))
            for asset_id in inventory.get_asset_ids()
        }


def _resolve(
    features: Mapping[str, dict],
    building_type_sources: Mapping[str, str],
    *,
    screen: ComposedScreen | None = None,
    vector_features: Sequence[str] = (),
) -> tuple[ResolutionResult, CompilationReport]:
    """
    Resolve an inventory built from feature dicts into a new report.

    Args:
        features (Mapping[str, dict]):
            Features by asset id.
        building_type_sources (Mapping[str, str]):
            The sources of the assets' building types.
        screen (ComposedScreen or None):
            The screen the loop tests against. ``None`` takes
            :func:`_screen`'s.
        vector_features (Sequence[str]):
            Names of the features whose values are vectors.

    Returns:
        tuple[ResolutionResult, CompilationReport]:
            The result and the report it was recorded in.
    """
    inventory = AssetInventory(vector_features=list(vector_features))
    for asset_id, asset in features.items():
        inventory.add_asset(asset_id, Asset(asset_id, [[-122.4, 37.8]], dict(asset)))
    report = CompilationReport(TYPE_SEED)
    loop = ResolutionLoop(_screen() if screen is None else screen)
    result = loop.resolve(
        inventory, report, building_type_sources=building_type_sources
    )
    return result, report


def test_an_nsi_type_the_cell_cannot_hold_is_withheld() -> None:
    """Withholding admits the asset, and nothing else about it changes."""
    features = dict(MASONRY_WALKUP)

    result, report = _resolve({'a': features}, {'a': 'nsi'})

    assert result.resolved_by_withholding == 1
    assert result.given_up == {'a': (BUILDING_TYPE_KEY,)}
    assert result.resolved.get_asset_features('a')[1] == {
        key: value
        for key, value in MASONRY_WALKUP.items()
        if key != BUILDING_TYPE_KEY
    }
    assert features == MASONRY_WALKUP
    assert report.records[LEDGER_COLUMNS].to_dict('records') == [
        {
            'asset_id': 'a',
            'world': WORLD,
            'stage': STAGE_RESOLUTION,
            'action': 'withheld',
            'feature': BUILDING_TYPE_KEY,
            'old': 'M',
            'new': None,
            'source': 'nsi',
            'block_group': BLOCK_GROUP,
        }
    ]


@pytest.mark.parametrize('type_source', ['local', 'footprint'])
def test_a_local_or_footprint_type_is_never_withheld(type_source: str) -> None:
    """A type the loop may not act on leaves the asset to its category."""
    screen = _screen()

    result, report = _resolve(
        {'a': MASONRY_WALKUP}, {'a': type_source}, screen=screen
    )

    assert result.excluded == ['a']
    assert report.records.empty
    exclusion = report.exclusions.iloc[0]
    assert exclusion['world'] == WORLD
    assert exclusion['stage'] == STAGE_STRUCTURE_TYPE
    assert exclusion['category'] == _category(
        screen, screen.screen(MASONRY_WALKUP), MASONRY_WALKUP, acted_on=False
    )


@pytest.mark.parametrize(
    'screen',
    [
        _screen(complete_tables=False),
        _ScriptedScreen(
            lambda _features: ScreenVerdict(
                DIAGNOSIS_INFEASIBLE, '', STAGE_STRUCTURE_TYPE, (OCCUPANCY_KEY,)
            )
        ),
    ],
    ids=('empty_cell', 'other_feature_named'),
)
def test_an_nsi_type_the_refusal_does_not_name_is_not_withheld(
    screen: ComposedScreen,
) -> None:
    """Only a type among the refusal's candidates is withheld."""
    result, report = _resolve({'a': TALL_WOOD_STORE}, {'a': 'nsi'}, screen=screen)

    assert result.excluded == ['a']
    assert report.records.empty
    exclusion = report.exclusions.iloc[0]
    assert exclusion['stage'] == STAGE_STRUCTURE_TYPE
    assert exclusion['category'] == _category(
        screen, screen.screen(TALL_WOOD_STORE), TALL_WOOD_STORE, acted_on=False
    )


def test_a_type_withholding_cannot_admit_is_restored_once() -> None:
    """An asset still refused is excluded as a conflict with its type put back."""
    refusal = ScreenVerdict(
        DIAGNOSIS_INFEASIBLE,
        'nothing to draw',
        STAGE_STRUCTURE_TYPE,
        (BUILDING_TYPE_KEY,),
    )
    screen = _ScriptedScreen(lambda _features: refusal)

    result, report = _resolve({'a': MASONRY_WALKUP}, {'a': 'nsi'}, screen=screen)

    assert result.excluded == ['a']
    records = report.records
    assert list(records['action']) == ['withheld', 'restored']
    assert records.iloc[0]['old'] == 'M'
    assert records.iloc[1]['new'] == 'M'
    exclusion = report.exclusions.iloc[0]
    assert exclusion['stage'] == STAGE_RESOLUTION
    assert exclusion['category'] == 'conflict'
    assert exclusion['detail'] == (
        'nothing to draw; without BuildingType: nothing to draw'
    )
    assert exclusion['resolvable_by'] == (BUILDING_TYPE_KEY,)


@pytest.mark.parametrize(
    ('features', 'stage', 'category'),
    [
        (
            {**MASONRY_WALKUP, YEAR_BUILT_KEY: [1945, 1955]},
            STAGE_DESIGN_LEVEL,
            'unsupported',
        ),
        (
            {**MASONRY_WALKUP, YEAR_BUILT_KEY: 'unknown'},
            STAGE_DESIGN_LEVEL,
            'missing_driver',
        ),
    ],
    ids=('per_world_list', 'unreadable_year'),
)
def test_an_asset_whose_drivers_cannot_be_read_takes_no_action(
    features: dict, stage: str, category: str
) -> None:
    """No type is withheld from an asset the chain cannot read."""
    _, report = _resolve({'a': features}, {'a': 'nsi'})

    assert report.records.empty
    exclusion = report.exclusions.iloc[0]
    assert exclusion['stage'] == stage
    assert exclusion['category'] == category


def test_an_unsupported_asset_takes_no_action_whatever_it_names() -> None:
    """An unsupported refusal is excluded at its stage, its named type kept."""
    refusal = ScreenVerdict(
        DIAGNOSIS_UNSUPPORTED,
        'a list per world',
        STAGE_STRUCTURE_TYPE,
        (BUILDING_TYPE_KEY,),
    )
    screen = _ScriptedScreen(lambda _features: refusal)

    result, report = _resolve({'a': MASONRY_WALKUP}, {'a': 'nsi'}, screen=screen)

    assert result.excluded == ['a']
    assert report.records.empty
    exclusion = report.exclusions.iloc[0]
    assert exclusion['stage'] == STAGE_STRUCTURE_TYPE
    assert exclusion['category'] == 'unsupported'


def test_a_declared_vector_feature_is_read_as_one_value() -> None:
    """A list under a vector feature is not taken for one value per world."""
    features = {**STORE, OCCUPANCY_KEY: ['COM1', 'COM1']}

    _, report = _resolve({'a': features}, {}, vector_features=[OCCUPANCY_KEY])

    assert report.exclusions.iloc[0]['category'] == 'missing_driver'


@pytest.mark.parametrize(
    ('features', 'building_type_sources'),
    [
        (MASONRY_WALKUP, {'a': 'osm'}),
        (STORE, {'a': 'nsi'}),
        (MASONRY_WALKUP, {}),
    ],
    ids=('unknown_label', 'no_type', 'no_entry'),
)
def test_a_building_type_source_that_cannot_be_read_is_refused(
    features: dict, building_type_sources: dict
) -> None:
    """Every type the asset holds has a source, given by its label."""
    with pytest.raises(ValueError, match='building_type_sources'):
        _resolve({'a': features}, building_type_sources)


def test_a_refused_input_leaves_the_report_as_it_was() -> None:
    """Every asset is checked before any is screened or recorded."""
    inventory = AssetInventory()
    for asset_id in ('a', 'b'):
        inventory.add_asset(
            asset_id, Asset(asset_id, [[-122.4, 37.8]], dict(MASONRY_WALKUP))
        )
    report = CompilationReport(TYPE_SEED)

    with pytest.raises(ValueError, match='building_type_sources'):
        ResolutionLoop(_screen()).resolve(
            inventory, report, building_type_sources={'a': 'nsi'}
        )

    assert report.records.empty
    assert report.exclusions.empty


def test_the_counts_agree_with_the_report() -> None:
    """Every asset is admitted or excluded once, and counted where it went."""
    result, report = _resolve(
        {
            'passes': STORE,
            'withheld': MASONRY_WALKUP,
            'local': MASONRY_WALKUP,
        },
        {'withheld': 'nsi', 'local': 'local'},
    )

    assert result.passed_first_screen == 1
    assert result.resolved_by_withholding == 1
    assert list(result.resolved.get_asset_ids()) == ['passes', 'withheld']
    assert result.excluded == ['local']
    assert result.excluded_count == len(report.exclusions)
    assert list(report.exclusions['world']) == [WORLD]


def test_the_order_of_the_inventory_changes_no_outcome() -> None:
    """Each asset is resolved from its own features alone."""
    features = {
        'passes': STORE,
        'withheld': MASONRY_WALKUP,
        'local': MASONRY_WALKUP,
    }
    sources = {'withheld': 'nsi', 'local': 'local'}

    forward, _ = _resolve(features, sources)
    backward, _ = _resolve(dict(reversed(features.items())), sources)

    assert forward.given_up == backward.given_up
    assert forward.excluded == backward.excluded
    assert {
        asset_id: forward.resolved.get_asset_features(asset_id)[1]
        for asset_id in forward.resolved.get_asset_ids()
    } == {
        asset_id: backward.resolved.get_asset_features(asset_id)[1]
        for asset_id in backward.resolved.get_asset_ids()
    }


def test_screen_discovered_and_exported() -> None:
    """The Importer finds the class, and the package exports it."""
    assert Importer().get_class('ComposedScreen') is ComposedScreen
    assert 'ComposedScreen' in brails.compilers.__all__
    assert brails.compilers.ComposedScreen is ComposedScreen


def test_loop_discovered_and_exported() -> None:
    """The Importer finds the class, and the package exports it."""
    assert Importer().get_class('ResolutionLoop') is ResolutionLoop
    assert 'ResolutionLoop' in brails.compilers.__all__
    assert brails.compilers.ResolutionLoop is ResolutionLoop
