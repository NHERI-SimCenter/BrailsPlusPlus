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

"""Tests for the composed screen."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import brails.compilers
from brails.compilers.compilation_report import EXCLUSION_CATEGORIES
from brails.compilers.resolution_loop import (
    SCREEN_ASSET_ID,
    STAGE_DESIGN_LEVEL,
    STAGE_STRUCTURE_TYPE,
    ComposedScreen,
    ScreenVerdict,
    _category,
)
from brails.constants import (
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
    diagnosis_frame,
)
from brails.utils import Importer

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from brails.types.asset_inventory import AssetInventory

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

# A wood high-rise. The tables print the cell, and no wood type in it.
WOOD_TOWER = {
    OCCUPANCY_KEY: 'RES3',
    STORIES_KEY: 9,
    YEAR_BUILT_KEY: 1960,
    BUILDING_TYPE_KEY: 'W',
}

# A wood store at eight stories. The tables print no row for the cell,
# and the row the completed tables fill it with holds no wood type.
TALL_WOOD_STORE = {**STORE, STORIES_KEY: 8, BUILDING_TYPE_KEY: 'W'}


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
            The screen.
    """
    settings = {
        'design_level_inferer': HazusEQDesignLevelInferer(
            state=STATE, seismic_zone=SEISMIC_ZONE
        ),
        'structure_type_inferer': HazusEQStructureTypeInferer(
            STATE, seed=TYPE_SEED, complete_tables=complete_tables
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


def test_screen_discovered_and_exported() -> None:
    """The Importer finds the class, and the package exports it."""
    assert Importer().get_class('ComposedScreen') is ComposedScreen
    assert 'ComposedScreen' in brails.compilers.__all__
    assert brails.compilers.ComposedScreen is ComposedScreen
