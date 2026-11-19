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


"""
Draw a Hazus structure type for every asset in an inventory.

.. autosummary::

      HazusEQStructureTypeInferer
"""

from __future__ import annotations

import secrets
from copy import deepcopy
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    UNREINFORCED_MASONRY_TYPES,
    WEST_COAST,
    get_hazus_general_type_membership,
    get_hazus_occ_type_mapping,
    get_hazus_sampling_height_band,
    get_hazus_sampling_height_bands,
    get_hazus_state_region_mapping,
    get_hazus_story_count,
    get_hazus_unreinforced_design_levels,
    get_hazus_year_classes,
)
from brails.inferers.hazus_inferer_earthquake.hazus_table_completion import (
    CellFill,
    get_completed_west_coast_weights,
    get_west_coast_fill_routes,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INFEASIBLE,
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
    diagnosis_frame,
    preview_failures,
)
from brails.inferers.inference_engine import InferenceEngine
from brails.types.asset_inventory import is_probabilistic_feature
from brails.utils.deterministic_sampling import (
    DEFAULT_ITERATION,
    DEFAULT_WORLD,
    asset_key_digest,
    grouped_categorical,
    keyed_uniforms,
)

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory

# Part of the reproducibility contract; see keyed_uniforms.
STRUCTURE_TYPE_PURPOSE = 'structure_type'

# Design levels the Hazus earthquake models define, in increasing order of
# seismic design strength. Every design level table starts at Pre-Code, so
# an early enough year always reaches a level that admits the unreinforced
# types.
DESIGN_LEVELS = (
    'Pre-Code',
    'Low-Code',
    'Moderate-Code',
    'High-Code',
    'Very High-Code',
    'Severe-Code',
)
EARLIEST_DESIGN_LEVEL = DESIGN_LEVELS[0]

# The multifamily classes the earthquake tables carry one RES3 row for. The
# replacement cost tables are keyed by subtype, so the fold is local to the
# structure type lookup and the asset keeps the class it arrived with.
MULTIFAMILY_OCCUPANCIES = (
    'RES3',
    'RES3A',
    'RES3B',
    'RES3C',
    'RES3D',
    'RES3E',
    'RES3F',
)

_SEED_BITS = 64


class _Verdict(NamedTuple):
    """
    What the inferer can do with one asset, and why.

    Attributes:
        diagnosis (str):
            One of the verdicts of the shared vocabulary.
        detail (str):
            What happened, in words.
        resolvable_by (tuple):
            Features that would each, alone, make the asset drawable.
        types (tuple):
            The structure types left to draw from.
        weights (tuple):
            Their weights, aligned position by position.
    """

    diagnosis: str
    detail: str = ''
    resolvable_by: tuple = ()
    types: tuple = ()
    weights: tuple = ()


class _Frame(NamedTuple):
    """
    The feature values that select a cell of the Appendix A tables.

    Attributes:
        occupancy (str):
            Occupancy class as the tables key it.
        band (str):
            Story band, empty for RES1, which has no height dimension.
        era (str):
            Construction era, empty outside the West Coast.
        general_type (str):
            Observed general building type, empty where none was read.
        design_level (str):
            The asset's design level.
    """

    occupancy: str
    band: str
    era: str
    general_type: str
    design_level: str


class HazusEQStructureTypeInferer(InferenceEngine):
    """
    Draw a Hazus structure type for every asset in an inventory.

    The Hazus Inventory Technical Manual tabulates the structure types of
    the general building stock as percentages, by region, occupancy
    class, story band, and, on the West Coast, construction era. This
    class draws one type per asset from the cell its features select.

    Two restrictions narrow that cell before the draw. An observed
    general building type keeps only the Hazus types belonging to it, and
    a design level outside the region's legality set removes S5, C3 and
    URM. What remains is renormalized, so a cell reduced to one type
    draws that type with certainty.

    Where the cell comes out empty, ``check`` reports which features
    would each, on their own, give the asset something to draw. It does
    not rank them: which of them to give up is the caller's decision,
    made against whatever trust ranking the caller holds.

    The draw belongs to the asset rather than to its position, so an
    inventory drawn in pieces and an inventory drawn whole give the same
    types. Each inferer draws its own seed unless given one; read
    ``self.seed`` to learn the value a run used, and pass it back to
    repeat that run.

    One state per inferer. An inventory spanning several states needs one
    inferer per state, each given the assets in it.
    """

    def __init__(
        self,
        state: str = 'California',
        seed: int | None = None,
        *,
        use_building_type: bool = True,
        unreinforced_design_levels: tuple[str, ...] | None = None,
        occupancy_key: str = 'OccupancyClass',
        stories_key: str = 'NumberOfStories',
        year_built_key: str = 'YearBuilt',
        design_level_key: str = 'DesignLevel',
        building_type_key: str = 'BuildingType',
        structure_type_key: str = 'StructureType',
        world: int = DEFAULT_WORLD,
        complete_tables: bool = False,
    ) -> None:
        """
        Set up the inferer and resolve its region tables.

        Args:
            state (str):
                Full state name, as in 'California'. It selects the
                regional table set through Table A-22 and, for RES1, the
                state's own row.
            seed (int | None):
                Seed for the draw. Left out, a seed is generated at
                random, so two runs of the same inventory give different
                types. The value used is recorded on ``self.seed``.
            use_building_type (bool):
                Whether an observed general building type restricts the
                draw to its own Hazus types.
            unreinforced_design_levels (tuple[str, ...] | None):
                Design levels that admit URM, C3 and S5, overriding the
                manual's set for this inferer's region. Left out, the
                manual governs: Pre-Code alone on the West Coast,
                Pre-Code and Low-Code elsewhere. Passing
                ``('Pre-Code', 'Low-Code')`` on the West Coast applies
                instead the nationwide bound the fragility library
                carries. What bears on that choice, including how much
                of the West Coast stock the regional clause reaches, is
                in the "Scope" section of the README beside this module.
            occupancy_key (str):
                Feature name the occupancy class is read from.
            stories_key (str):
                Feature name the story count is read from.
            year_built_key (str):
                Feature name the construction year is read from. Only
                the West Coast tables need it.
            design_level_key (str):
                Feature name the design level is read from.
            building_type_key (str):
                Feature name the general building type is read from.
            structure_type_key (str):
                Feature name the drawn type is written to.
            world (int):
                Possible-world index. Only a single world is supported
                currently.
            complete_tables (bool):
                Whether to draw the West Coast cells no manual prints a
                row for from the completed tables of
                :mod:`.hazus_table_completion`, whose
                ``get_west_coast_fill_routes`` states how each cell is
                filled. With the default, an asset
                in such a cell is infeasible. Printed cells draw as
                printed either way. Whether an inventory should draw
                from the completion is the caller's decision.

        Raises:
            ValueError:
                If the state is not one the Hazus region table names, if
                the tables carry no RES1 distribution for it, if the
                legality set given is one no year could reach, or if
                ``complete_tables`` is set outside the West Coast.

                A state without a RES1 distribution is refused here because
                an inventory compiled that way would silently hold no
                single-family homes at all.
        """
        regions = get_hazus_state_region_mapping()
        if state not in regions:
            raise ValueError(
                f'{state!r} is not a state in the Hazus region table. Give '
                "the full name, as in 'California'; two-letter codes and "
                'abbreviations are not accepted.'
            )
        type_lists, type_weights = get_hazus_occ_type_mapping()
        self.state = state
        self.region = regions[state]['RegionGroup']
        self._type_lists = type_lists[self.region]
        self._weights = type_weights[self.region]
        # Only the West Coast tables are stratified by era.
        self._needs_era = self.region == WEST_COAST
        self._check_single_family_coverage()
        self.seed = secrets.randbits(_SEED_BITS) if seed is None else int(seed)
        self.use_building_type = bool(use_building_type)
        self.occupancy_key = occupancy_key
        self.stories_key = stories_key
        self.year_built_key = year_built_key
        self.design_level_key = design_level_key
        self.building_type_key = building_type_key
        self.structure_type_key = structure_type_key
        self.world = int(world)
        self._unreinforced_design_levels = self._legality_set(
            unreinforced_design_levels
        )
        self._bands = get_hazus_sampling_height_bands()
        self._membership = get_hazus_general_type_membership()
        self._era_by_year = {
            year: era
            for era, years in get_hazus_year_classes().items()
            for year in years
        }
        self.complete_tables = bool(complete_tables)
        self._completed_weights = None
        self._fill_routes = {}
        if self.complete_tables:
            if self.region != WEST_COAST:
                raise ValueError(
                    f'complete_tables applies to the {WEST_COAST} tables '
                    f'only. {self.state} draws from the {self.region} '
                    'tables.'
                )
            self._completed_weights = get_completed_west_coast_weights()
            self._fill_routes = get_west_coast_fill_routes()

    def _legality_set(self, override: tuple[str, ...] | None) -> tuple:
        """
        Settle which design levels admit the unreinforced types.

        Args:
            override (tuple[str, ...] | None):
                The caller's set, or None to take the manual's.

        Returns:
            tuple:
                The design levels that admit them in this region.

        Raises:
            ValueError:
                If the override names something that is not a design
                level, or leaves out the earliest one.
        """
        if override is None:
            return get_hazus_unreinforced_design_levels()[self.region]
        levels = tuple(override)
        unknown = [level for level in levels if level not in DESIGN_LEVELS]
        if unknown:
            raise ValueError(
                f'{unknown} are not Hazus design levels. Valid levels: '
                f'{list(DESIGN_LEVELS)}.'
            )
        if EARLIEST_DESIGN_LEVEL not in levels:
            raise ValueError(
                f'A legality set must admit {EARLIEST_DESIGN_LEVEL}, which '
                f'{levels} does not. Every design level table begins there, '
                'so a set without it leaves buildings whose year can be '
                'moved no design level that admits the unreinforced types, '
                'and no year would resolve them.'
            )
        return levels

    def _check_single_family_coverage(self) -> None:
        """
        Refuse a state the tables cannot produce single-family homes for.

        Raises:
            ValueError:
                If any era's RES1 table lacks a row for the state.
        """
        rows = self._weights['RES1']
        by_era = list(rows.values()) if self._needs_era else [rows]
        if all(self.state in row for row in by_era):
            return
        served = set.intersection(*(set(row) for row in by_era))
        missing = sorted(
            state
            for state, entry in get_hazus_state_region_mapping().items()
            if entry['RegionGroup'] == self.region and state not in served
        )
        raise ValueError(
            f'The Appendix A tables carry no RES1 distribution for '
            f'{self.state}, so no single-family home there can be drawn. '
            f'The states this affects: {missing}.'
        )

    def check(self, input_inventory: AssetInventory) -> pd.DataFrame:
        """
        Diagnose every asset without drawing anything.

        Args:
            input_inventory (AssetInventory):
                The inventory to diagnose.

        Returns:
            pd.DataFrame:
                One row per asset id, with the columns of
                ``DIAGNOSIS_COLUMNS``. An infeasible asset carries in
                ``resolvable_by`` every feature that would, on its own,
                give it something to draw. Their order is fixed and
                carries no preference: which feature to give up is the
                caller's decision.
        """
        rows = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            verdict = self._resolve(features, input_inventory.vector_features)
            rows[asset_id] = (
                verdict.diagnosis,
                verdict.detail,
                verdict.resolvable_by,
            )
        return diagnosis_frame(rows)

    def draw(
        self,
        input_inventory: AssetInventory,
        *,
        iteration: int = DEFAULT_ITERATION,
        asset_keys: np.ndarray | None = None,
    ) -> pd.Series:
        """
        Draw a structure type for every asset, assuming all are usable.

        Args:
            input_inventory (AssetInventory):
                An inventory whose assets all passed :meth:`check`.
            iteration (int):
                Resolution-loop attempt number. Distinct iterations draw
                independently for the same asset.
            asset_keys (np.ndarray | None):
                Keys from :func:`asset_key_digest`, in the order
                ``get_asset_ids`` returns them. Digesting dominates the
                cost of a redraw, so a caller redrawing the same assets
                should digest once and pass the keys in.

        Returns:
            pd.Series:
                Structure types indexed by asset id.

        Raises:
            ValueError:
                If an asset cannot be drawn for, which means
                :meth:`check` was skipped, or if the keys given do not
                match the assets.
        """
        asset_ids = list(input_inventory.get_asset_ids())
        if not asset_ids:
            return pd.Series(dtype=object, name=self.structure_type_key)
        vector_features = input_inventory.vector_features
        signatures: dict[tuple, int] = {}
        codes = []
        for asset_id in asset_ids:
            _, features = input_inventory.get_asset_features(asset_id)
            verdict = self._resolve(features, vector_features)
            if verdict.diagnosis != DIAGNOSIS_OK:
                raise ValueError(
                    f'Cannot draw {self.structure_type_key} for asset '
                    f'{asset_id}: {verdict.detail}. Screen the inventory '
                    'with check() before calling draw().'
                )
            signature = (verdict.types, verdict.weights)
            codes.append(signatures.setdefault(signature, len(signatures)))
        if asset_keys is None:
            keys = asset_key_digest(asset_ids)
        else:
            keys = np.asarray(asset_keys)
            if keys.dtype != np.uint64 or keys.shape != (len(asset_ids),):
                raise ValueError(
                    f'Expected one key per asset from asset_key_digest, got '
                    f'an array of shape {keys.shape} and dtype {keys.dtype} '
                    f'for {len(asset_ids)} assets.'
                )
        uniforms = keyed_uniforms(
            keys,
            seed=self.seed,
            purpose=STRUCTURE_TYPE_PURPOSE,
            world=self.world,
            iteration=iteration,
        )
        positions = grouped_categorical(
            uniforms,
            weights=[weights for _, weights in signatures],
            groups=np.array(codes, dtype=np.int64),
        )
        type_rows = [types for types, _ in signatures]
        drawn = [
            type_rows[code][position] for code, position in zip(codes, positions)
        ]
        return pd.Series(drawn, index=asset_ids, name=self.structure_type_key)

    def infer(self, input_inventory: AssetInventory) -> AssetInventory:
        """
        Draw a structure type for every asset in the inventory.

        Args:
            input_inventory (AssetInventory):
                The inventory to process. Every asset must carry usable
                drivers; screen the inventory with :meth:`check` first if
                that is not guaranteed.

        Returns:
            AssetInventory:
                A copy of the input with the drawn type added under
                ``structure_type_key``.

        Raises:
            ValueError:
                If any asset cannot be drawn for, naming the assets and
                their diagnoses.
        """
        verdicts = self.check(input_inventory)
        failed = verdicts[verdicts['diagnosis'] != DIAGNOSIS_OK]
        if not failed.empty:
            raise ValueError(
                f'Cannot draw {self.structure_type_key} for {len(failed)} of '
                f'{len(verdicts)} assets: {preview_failures(failed["detail"])}'
                '. Remove them or supply the missing data, then call infer() '
                'again.'
            )
        types = self.draw(input_inventory)
        output_inventory = deepcopy(input_inventory)
        for asset_id, structure_type in types.items():
            output_inventory.add_asset_features(
                asset_id, {self.structure_type_key: structure_type}
            )
        return output_inventory

    def _resolve(self, features: dict, vector_features: set) -> _Verdict:
        """
        Work out what an asset can draw from, or why it cannot.

        Args:
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors, where a
                list is one value and not one value per world.

        Returns:
            _Verdict:
                The candidate types and weights, or a diagnosis.
        """
        frame, failure = self._read_frame(features, vector_features)
        if failure is not None:
            return failure
        types, weights = self._candidates(frame)
        if sum(weights) > 0:
            return _Verdict(DIAGNOSIS_OK, types=tuple(types), weights=tuple(weights))
        return _Verdict(
            DIAGNOSIS_INFEASIBLE,
            self._describe(frame),
            self._resolvable_by(frame),
        )

    def _read_frame(
        self, features: dict, vector_features: set
    ) -> tuple[_Frame | None, _Verdict | None]:
        """
        Read the features that select a cell, refusing what cannot be used.

        Args:
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors.

        Returns:
            tuple[_Frame | None, _Verdict | None]:
                The frame and None, or None and the verdict that refuses
                the asset.
        """
        occupancy, failure = self._read(
            features, self.occupancy_key, vector_features
        )
        if failure is not None:
            return None, failure
        occupancy = str(occupancy)
        table_occupancy = (
            'RES3' if occupancy in MULTIFAMILY_OCCUPANCIES else occupancy
        )
        if table_occupancy not in self._weights:
            return None, _Verdict(
                DIAGNOSIS_INVALID,
                f'{self.occupancy_key} {occupancy!r} has no row in the '
                f'{self.region} tables',
            )
        design_level, failure = self._read(
            features, self.design_level_key, vector_features
        )
        if failure is not None:
            return None, failure
        design_level = str(design_level)
        if design_level not in DESIGN_LEVELS:
            return None, _Verdict(
                DIAGNOSIS_INVALID,
                f'{self.design_level_key} {design_level!r} is not a Hazus '
                'design level',
            )
        era = ''
        if self._needs_era:
            era, failure = self._read_era(features, vector_features)
            if failure is not None:
                return None, failure
        band = ''
        if table_occupancy == 'RES1':
            # RES1 is tabulated without a story band, so a count is not
            # required here. It is still read, because a change of occupancy
            # would carry the asset into whichever band the count gives it,
            # and an unusable count simply leaves that path unavailable.
            band, _ = self._read_band(features, vector_features)
        else:
            band, failure = self._read_band(features, vector_features)
            if failure is not None:
                return None, failure
        general_type, failure = self._read_general_type(features, vector_features)
        if failure is not None:
            return None, failure
        return (
            _Frame(table_occupancy, band, era, general_type, design_level),
            None,
        )

    def _read(
        self, features: dict, key: str, vector_features: set
    ) -> tuple[object | None, _Verdict | None]:
        """
        Read one feature, refusing what cannot be used.

        Args:
            features (dict):
                The asset's features.
            key (str):
                The feature name to read.
            vector_features (set):
                Names of the features whose values are vectors.

        Returns:
            tuple[object | None, _Verdict | None]:
                The value and None, or None and a verdict.
        """
        value = features.get(key)
        if value is None:
            return None, _Verdict(DIAGNOSIS_INVALID, f'no {key}')
        if is_probabilistic_feature(key, value, vector_features):
            return None, _Verdict(
                DIAGNOSIS_UNSUPPORTED,
                f'{key} holds one value per world; multi-world drawing is '
                'planned and not supported yet',
            )
        return value, None

    def _read_era(
        self, features: dict, vector_features: set
    ) -> tuple[str, _Verdict | None]:
        """
        Select the construction era of an asset.

        Args:
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors.

        Returns:
            tuple[str, _Verdict | None]:
                The era and None, or an empty string and a verdict.
        """
        year, failure = self._read(features, self.year_built_key, vector_features)
        if failure is not None:
            return '', failure
        try:
            year_value = float(year)
        except (TypeError, ValueError):
            return '', _Verdict(
                DIAGNOSIS_INVALID,
                f'{self.year_built_key} is not a number: {year!r}',
            )
        if not np.isfinite(year_value) or year_value != int(year_value):
            return '', _Verdict(
                DIAGNOSIS_INVALID,
                f'{self.year_built_key} of {year!r} is not a whole year',
            )
        era = self._era_by_year.get(int(year_value))
        if era is None:
            return '', _Verdict(
                DIAGNOSIS_INVALID,
                f'{self.year_built_key} of {year!r} sits outside the years '
                f'the {self.region} tables cover',
            )
        return era, None

    def _read_band(
        self, features: dict, vector_features: set
    ) -> tuple[str, _Verdict | None]:
        """
        Select the story band of an asset.

        Args:
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors.

        Returns:
            tuple[str, _Verdict | None]:
                The band and None, or an empty string and a verdict.
        """
        stories, failure = self._read(features, self.stories_key, vector_features)
        if failure is not None:
            return '', failure
        try:
            story_count = get_hazus_story_count(stories)
        except TypeError:
            return '', _Verdict(
                DIAGNOSIS_INVALID,
                f'{self.stories_key} is not a number: {stories!r}',
            )
        except ValueError:
            return '', _Verdict(
                DIAGNOSIS_INVALID,
                f'{self.stories_key} of {stories!r} is not a whole number of '
                'stories of one or more',
            )
        return self._band_of(story_count), None

    def _read_general_type(
        self, features: dict, vector_features: set
    ) -> tuple[str, _Verdict | None]:
        """
        Read the observed general building type, which is optional.

        Args:
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors.

        Returns:
            tuple[str, _Verdict | None]:
                The general type and None, empty where none was observed
                or the inferer was told to ignore it, or a verdict.
        """
        if not self.use_building_type:
            return '', None
        value = features.get(self.building_type_key)
        if value is None:
            return '', None
        if is_probabilistic_feature(self.building_type_key, value, vector_features):
            return '', _Verdict(
                DIAGNOSIS_UNSUPPORTED,
                f'{self.building_type_key} holds one value per world; '
                'multi-world drawing is planned and not supported yet',
            )
        general_type = str(value)
        if general_type not in self._membership:
            return '', _Verdict(
                DIAGNOSIS_INVALID,
                f'{self.building_type_key} {general_type!r} is not a Hazus '
                'general building type',
            )
        return general_type, None

    def _band_of(self, stories: int) -> str:
        """
        Select the story band a building falls in.

        Args:
            stories (int):
                Story count, one or more.

        Returns:
            str:
                The band name.
        """
        return get_hazus_sampling_height_band(stories)

    def _candidates(self, frame: _Frame) -> tuple[tuple, tuple]:
        """
        Return the types and weights a frame leaves to draw from.

        Args:
            frame (_Frame):
                The feature values that select the cell.

        Returns:
            tuple[tuple, tuple]:
                Structure types and their weights, both empty where the
                frame selects no row at all.
        """
        # completed weights replace the printed row except for RES1
        if self._completed_weights is None or frame.occupancy == 'RES1':
            types, row = self._published_row(frame)
        else:
            types = self._type_lists[frame.band]
            row = self._completed_weights[frame.occupancy][frame.band][frame.era]
        if not row:
            return (), ()
        members = self._membership.get(frame.general_type)
        legal = frame.design_level in self._unreinforced_design_levels
        kept = [
            (structure_type, float(weight))
            for structure_type, weight in zip(types, row)
            if (members is None or structure_type in members)
            and (legal or structure_type not in UNREINFORCED_MASONRY_TYPES)
        ]
        return (
            tuple(structure_type for structure_type, _ in kept),
            tuple(weight for _, weight in kept),
        )

    def _published_row(self, frame: _Frame) -> tuple[tuple, tuple]:
        """
        Return the types and weights the tables print for a frame's cell.

        Args:
            frame (_Frame):
                The feature values that select the cell.

        Returns:
            tuple[tuple, tuple]:
                The structure types of the cell and the weights printed
                against them, before any restriction narrows them. The
                weights are empty where the tables print no such row.
        """
        if frame.occupancy == 'RES1':
            rows = self._weights['RES1']
            row = (rows[frame.era] if self._needs_era else rows)[self.state]
            return self._type_lists['RES1'], row
        cell = self._weights[frame.occupancy][frame.band]
        row = cell[frame.era] if self._needs_era else cell
        return self._type_lists[frame.band], row

    def _frame_or_raise(
        self,
        features: dict,
        vector_features: frozenset[str] | None,
        question: str,
    ) -> _Frame:
        """
        Read the cell an asset's features select, raising where none is.

        Args:
            features (dict):
                One asset's features.
            vector_features (frozenset[str] or None):
                Names of the features whose values are vectors.
            question (str):
                What the caller asked about the cell, for the message.

        Returns:
            _Frame:
                The feature values that select the cell.

        Raises:
            ValueError:
                If a driver cannot be read, since the features then
                select no cell.
        """
        frame, failure = self._read_frame(features, vector_features or frozenset())
        if failure is not None:
            raise ValueError(
                f'The features select no cell of the {self.region} tables: '
                f'{failure.detail}. Diagnose the asset with check() before '
                f'asking {question}.'
            )
        return frame

    def cell_is_unpublished(
        self,
        features: dict,
        *,
        vector_features: frozenset[str] | None = None,
    ) -> bool:
        """
        Say whether the tables print no distribution for an asset's cell.

        An asset the tables have no row for and an asset whose row the
        restrictions emptied both come back infeasible from
        :meth:`check`, and the two call for different answers: the first
        is a gap in the published model and the second is a disagreement
        among the asset's own features.

        Args:
            features (dict):
                One asset's features.
            vector_features (frozenset[str] or None):
                Names of the features whose values are vectors.

        Returns:
            bool:
                True where the cell the features select carries no
                printed row.

        Raises:
            ValueError:
                If a driver cannot be read, since the features then
                select no cell.
        """
        frame = self._frame_or_raise(
            features, vector_features, 'whether its cell is published'
        )
        _, row = self._published_row(frame)
        return not row

    def cell_fill(
        self,
        features: dict,
        *,
        vector_features: frozenset[str] | None = None,
    ) -> CellFill | None:
        """
        Say how the completed tables fill an asset's cell.

        With ``complete_tables`` set, :meth:`cell_is_unpublished` still
        reports a filled cell as unprinted. For such an asset found
        infeasible, this method shows that it had a filled row to draw
        from, so the infeasibility comes from its own features.

        Args:
            features (dict):
                One asset's features.
            vector_features (frozenset[str] or None):
                Names of the features whose values are vectors.

        Returns:
            CellFill or None:
                The fill the inferer draws the cell from, or None where
                no fill applies: a printed cell, or any cell when
                ``complete_tables`` is not set.

        Raises:
            ValueError:
                If a driver cannot be read, since the features then
                select no cell.
        """
        frame = self._frame_or_raise(
            features, vector_features, 'how its cell is filled'
        )
        return self._fill_routes.get((frame.occupancy, frame.band, frame.era))

    def _drawable(self, frame: _Frame) -> bool:
        """
        Say whether a frame would leave anything to draw.

        Args:
            frame (_Frame):
                The feature values that select the cell.

        Returns:
            bool:
                True where at least one type carries weight.
        """
        return sum(self._candidates(frame)[1]) > 0

    def _resolvable_by(self, frame: _Frame) -> tuple:
        """
        Name the features that would each, alone, make an asset drawable.

        Every feature is varied with the others held where the asset has
        them, so a name here is one the caller can act on by itself. The
        order is fixed and carries no preference.

        Args:
            frame (_Frame):
                The feature values that select the cell.

        Returns:
            tuple:
                Feature keys, empty where no single change would help.
        """
        resolvable = []
        if self._occupancy_resolves(frame):
            resolvable.append(self.occupancy_key)
        if frame.band and any(
            self._drawable(frame._replace(band=band))
            for band in self._band_names()
            if band != frame.band
        ):
            resolvable.append(self.stories_key)
        if self._year_resolves(frame):
            resolvable.append(self.year_built_key)
        if frame.general_type and self._drawable(frame._replace(general_type='')):
            resolvable.append(self.building_type_key)
        return tuple(resolvable)

    def _occupancy_resolves(self, frame: _Frame) -> bool:
        """
        Say whether some other occupancy class would admit the asset.

        A change of occupancy carries a RES1 asset into a story band, so
        it resolves nothing where the asset has no usable story count.

        Args:
            frame (_Frame):
                The feature values that select the cell.

        Returns:
            bool:
                True where some other occupancy would leave a type.
        """
        if frame.occupancy != 'RES1' and self._drawable(
            frame._replace(occupancy='RES1', band='')
        ):
            return True
        if not frame.band:
            return False
        return any(
            self._drawable(frame._replace(occupancy=occupancy))
            for occupancy in self._weights
            if occupancy not in (frame.occupancy, 'RES1')
        )

    def _year_resolves(self, frame: _Frame) -> bool:
        """
        Say whether some other construction year would admit the asset.

        The year moves two things: on the West Coast it selects the era,
        and everywhere it sets the design level. Every design level table
        begins at Pre-Code, which every region admits, so an early enough
        year always reaches an admitting design level; what it cannot
        escape on the West Coast is that the same year lands the asset in
        the earliest era.

        Args:
            frame (_Frame):
                The feature values that select the cell.

        Returns:
            bool:
                True where some year would leave a type.
        """
        if any(
            self._drawable(frame._replace(era=era))
            for era in self._eras()
            if era != frame.era
        ):
            return True
        earliest = self._eras()[0] if self._needs_era else ''
        return self._drawable(
            frame._replace(era=earliest, design_level=EARLIEST_DESIGN_LEVEL)
        )

    def _eras(self) -> tuple:
        """
        Return the construction eras the region's tables carry.

        Returns:
            tuple:
                Era names on the West Coast, oldest first; a single empty
                name elsewhere.
        """
        if not self._needs_era:
            return ('',)
        return tuple(self._weights['RES1'])

    def _band_names(self) -> tuple:
        """
        Return the story bands the non-RES1 tables carry.

        Returns:
            tuple:
                Band names.
        """
        return tuple(self._bands)

    def _describe(self, frame: _Frame) -> str:
        """
        State which feature values selected an empty cell, blaming none.

        Args:
            frame (_Frame):
                The feature values that select the cell.

        Returns:
            str:
                The message.
        """
        stated = [f'{self.occupancy_key} {frame.occupancy}']
        if frame.band:
            stated.append(f'{self.stories_key} in the {frame.band} band')
        if frame.era:
            stated.append(f'{self.year_built_key} in the {frame.era} era')
        if frame.general_type:
            stated.append(f'{self.building_type_key} {frame.general_type}')
        stated.append(f'{self.design_level_key} {frame.design_level}')
        return f'the {self.region} tables give no structure type for ' + ', '.join(
            stated
        )
