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

from brails.inferers.hazus_inferer_earthquake.hazus_eq_diagnosis import (
    RESTRICTION_BUILDING_TYPE,
    RESTRICTION_DESIGN_LEVEL,
    diagnosis_frame,
)
from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    get_hazus_general_type_membership,
    get_hazus_occ_type_mapping,
    get_hazus_sampling_height_bands,
    get_hazus_state_region_mapping,
    get_hazus_unreinforced_design_levels,
    get_hazus_year_classes,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_OK,
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

# The types the Appendix A footnote calls unreinforced. Which design
# levels admit them depends on the region; see
# get_hazus_unreinforced_design_levels.
UNREINFORCED_MASONRY_TYPES = ('S5', 'C3', 'URM')

# Design levels the Hazus earthquake models define, in increasing order
# of seismic design strength.
DESIGN_LEVELS = (
    'Pre-Code',
    'Low-Code',
    'Moderate-Code',
    'High-Code',
    'Very High-Code',
    'Severe-Code',
)

_SEED_BITS = 64


class _Cell(NamedTuple):
    """
    The outcome of resolving one asset against the Appendix A tables.

    Attributes:
        diagnosis (str):
            ``'OK'`` or what stands in the way of a draw.
        restriction (str):
            Which restriction emptied the candidate set, if one did.
        conflict (str):
            The two features that disagree, if a pair does.
        types (tuple):
            The structure types left to draw from.
        weights (tuple):
            Their weights, aligned position by position.
    """

    diagnosis: str
    restriction: str = ''
    conflict: str = ''
    types: tuple = ()
    weights: tuple = ()




class HazusEQStructureTypeInferer(InferenceEngine):
    """
    Draw a Hazus structure type for every asset in an inventory.

    The Hazus Inventory Technical Manual tabulates the structure types of
    the general building stock as percentages, by region, occupancy
    class, story band, and, on the West Coast, construction era. This
    class draws one type per asset from the cell its features select.

    Two restrictions narrow that cell before the draw. An observed
    general building type keeps only the Hazus types belonging to it, and
    a design level above Low-Code removes S5, C3 and URM, which have no
    archetype there. The remaining weights are renormalized, so a cell reduced
    to one type draws that type with certainty.

    The draw belongs to the asset, and it is not influenced by its position. An
    inventory drawn in pieces and an inventory drawn whole with the same seed
    give the same types. Each inferer draws its own seed unless given one; read
    ``self.seed`` to learn the value a run used, and pass it back to
    repeat that run.

    The current implementation is limited to one state per inferer. An
    inventory spanning several states needs one inferer per state.
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
                Pre-Code and Low-Code elsewhere.
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

        Raises:
            ValueError:
                If the state is not one the Hazus region table names.
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
        self.seed = secrets.randbits(_SEED_BITS) if seed is None else int(seed)
        self.use_building_type = bool(use_building_type)
        self.occupancy_key = occupancy_key
        self.stories_key = stories_key
        self.year_built_key = year_built_key
        self.design_level_key = design_level_key
        self.building_type_key = building_type_key
        self.structure_type_key = structure_type_key
        self.world = int(world)
        self._type_lists = type_lists[self.region]
        self._weights = type_weights[self.region]
        self._bands = get_hazus_sampling_height_bands()
        self._membership = get_hazus_general_type_membership()
        self._unreinforced_design_levels = (
            get_hazus_unreinforced_design_levels()[self.region]
            if unreinforced_design_levels is None
            else tuple(unreinforced_design_levels)
        )
        self._era_by_year = {
            year: era
            for era, years in get_hazus_year_classes().items()
            for year in years
        }
        # Only the West Coast tables are stratified by era.
        self._needs_era = self.region == 'West Coast'

    def check(self, input_inventory: AssetInventory) -> pd.DataFrame:
        """
        Diagnose every asset without drawing anything.

        Args:
            input_inventory (AssetInventory):
                The inventory to diagnose.

        Returns:
            pd.DataFrame:
                One row per asset id, with the columns of
                ``DIAGNOSIS_COLUMNS``. ``diagnosis`` is ``'OK'``
                where a type can be drawn and otherwise says what stands
                in the way. Where a restriction emptied the candidate
                set, ``restriction`` names it and ``conflict`` names the
                two features that disagree. The pair carries no order of
                precedence; a caller resolving the conflict decides which
                of the two to give up.
        """
        rows = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            cell = self._resolve(features, input_inventory.vector_features)
            rows[asset_id] = (cell.diagnosis, cell.restriction, cell.conflict)
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
            cell = self._resolve(features, vector_features)
            if cell.diagnosis != DIAGNOSIS_OK:
                raise ValueError(
                    f'Cannot draw {self.structure_type_key} for asset '
                    f'{asset_id}: {cell.diagnosis}. Screen the inventory '
                    'with check() before calling draw().'
                )
            signature = (cell.types, cell.weights)
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
            type_rows[code][position]
            for code, position in zip(codes, positions)
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
            reasons = preview_failures(failed['diagnosis'])
            raise ValueError(
                f'Cannot draw {self.structure_type_key} for {len(failed)} of '
                f'{len(verdicts)} assets: {reasons}. Remove them or supply '
                'the missing data, then call infer() again.'
            )
        types = self.draw(input_inventory)
        output_inventory = deepcopy(input_inventory)
        for asset_id, structure_type in types.items():
            output_inventory.add_asset_features(
                asset_id, {self.structure_type_key: structure_type}
            )
        return output_inventory

    def _resolve(self, features: dict, vector_features: set) -> _Cell:
        """
        Work out what an asset can draw from, or why it cannot.

        Args:
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors, where a
                list is one value and not one value per world.

        Returns:
            _Cell:
                The candidate types and weights, or a diagnosis.
        """
        occupancy, failure = self._read(
            features, self.occupancy_key, vector_features
        )
        if failure:
            return _Cell(failure)
        occupancy = str(occupancy)
        # The earthquake tables carry one RES3 row for all six multifamily
        # subtypes, while the replacement cost tables are keyed by subtype,
        # so the fold happens here and the asset keeps what it arrived with.
        table_occupancy = (
            'RES3' if occupancy.startswith('RES3') else occupancy
        )
        if table_occupancy not in self._weights:
            return _Cell(
                f'{self.occupancy_key} {occupancy!r} has no row in the '
                f'{self.region} tables'
            )
        design_level, failure = self._read(
            features, self.design_level_key, vector_features
        )
        if failure:
            return _Cell(failure)
        design_level = str(design_level)
        if design_level not in DESIGN_LEVELS:
            return _Cell(
                f'{self.design_level_key} {design_level!r} is not a Hazus '
                'design level'
            )
        era = ''
        if self._needs_era:
            era, failure = self._era(features, vector_features)
            if failure:
                return _Cell(failure)
        band = ''
        if table_occupancy == 'RES1':
            cell, failure = self._single_family_cell(era)
        else:
            cell, band, failure = self._general_cell(
                table_occupancy, era, features, vector_features
            )
        if failure:
            return _Cell(failure)
        return self._restrict(
            cell,
            table_occupancy,
            band,
            era,
            design_level,
            features,
            vector_features,
        )

    def _read(
        self, features: dict, key: str, vector_features: set
    ) -> tuple[object | None, str]:
        """
        Read one feature, refusing what cannot be used.

        Args:
            features (dict):
                The asset's features.
            key (str):
                The feature name to read.
            vector_features (set):
                Names of the features whose values are vectors, where a
                list is one value and not one value per world.

        Returns:
            tuple[object | None, str]:
                The value and an empty string, or None and a diagnosis.
        """
        value = features.get(key)
        if value is None:
            return None, f'no {key}'
        if is_probabilistic_feature(key, value, vector_features):
            return None, (
                f'{key} is a per-world list; multi-world drawing is not '
                'supported yet'
            )
        return value, ''

    def _era(
        self, features: dict, vector_features: set
    ) -> tuple[str, str]:
        """
        Select the construction era of an asset.

        Args:
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors, where a
                list is one value and not one value per world.

        Returns:
            tuple[str, str]:
                The era and an empty string, or an empty string and a
                diagnosis.
        """
        year, failure = self._read(
            features, self.year_built_key, vector_features
        )
        if failure:
            return '', failure
        try:
            year_value = float(year)
        except (TypeError, ValueError):
            return '', f'{self.year_built_key} is not a number: {year!r}'
        if not np.isfinite(year_value) or year_value != int(year_value):
            return '', (
                f'{self.year_built_key} of {year!r} is not a whole year'
            )
        era = self._era_by_year.get(int(year_value))
        if era is None:
            return '', (
                f'{self.year_built_key} of {year!r} sits outside the years '
                f'the {self.region} tables cover'
            )
        return era, ''

    def _single_family_cell(self, era: str) -> tuple[_Cell, str]:
        """
        Select the RES1 row for this inferer's state.

        Single-family homes are tabulated by state, with no story
        dimension, and their rows carry their own structure type list.

        Args:
            era (str):
                Construction era, empty outside the West Coast.

        Returns:
            tuple[_Cell, str]:
                The unrestricted cell and an empty string, or an empty
                cell and a diagnosis.
        """
        rows = self._weights['RES1']
        if self._needs_era:
            rows = rows[era]
        weights = rows.get(self.state)
        if weights is None:
            return _Cell(''), (
                f'{self.state} has no RES1 row in the {self.region} tables'
            )
        types = self._type_lists['RES1']
        if sum(weights) <= 0:
            return _Cell(''), (
                f'the {self.region} tables give no structure type for RES1 '
                f'in {self.state}'
            )
        return _Cell(DIAGNOSIS_OK, types=tuple(types),
                     weights=tuple(float(weight) for weight in weights)), ''

    def _general_cell(
        self,
        table_occupancy: str,
        era: str,
        features: dict,
        vector_features: set,
    ) -> tuple[_Cell, str, str]:
        """
        Select the occupancy and story band row for an asset.

        Args:
            table_occupancy (str):
                Occupancy class as the tables key it.
            era (str):
                Construction era, empty outside the West Coast.
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors, where a
                list is one value and not one value per world.

        Returns:
            tuple[_Cell, str, str]:
                The unrestricted cell, the story band it came from, and
                an empty string, or an empty cell and a diagnosis.
        """
        stories, failure = self._read(
            features, self.stories_key, vector_features
        )
        if failure:
            return _Cell(''), '', failure
        try:
            story_count = float(stories)
        except (TypeError, ValueError):
            return _Cell(''), '', (
                f'{self.stories_key} is not a number: {stories!r}'
            )
        if (
            not np.isfinite(story_count)
            or story_count != int(story_count)
            or story_count < 1
        ):
            return _Cell(''), '', (
                f'{self.stories_key} of {stories!r} is not a whole number of '
                'stories of one or more'
            )
        band = self._band(int(story_count))
        weights = self._weights[table_occupancy][band]
        if self._needs_era:
            weights = weights[era]
        types = self._type_lists[band]
        if not weights or sum(weights) <= 0:
            return _Cell(''), band, (
                f'the {self.region} tables give no structure type for '
                f'{self.occupancy_key} {table_occupancy} at '
                f'{int(story_count)} stories'
            )
        return (
            _Cell(
                DIAGNOSIS_OK,
                types=tuple(types),
                weights=tuple(float(weight) for weight in weights),
            ),
            band,
            '',
        )

    def _band(self, stories: int) -> str:
        """
        Select the story band a building falls in.

        Args:
            stories (int):
                Story count, one or more.

        Returns:
            str:
                The band name.
        """
        for band, (lowest, highest) in self._bands.items():
            if stories >= lowest and (highest is None or stories <= highest):
                return band
        raise ValueError(f'No sampling band covers {stories} stories.')

    def _restrict(
        self,
        cell: _Cell,
        table_occupancy: str,
        band: str,
        era: str,
        design_level: str,
        features: dict,
        vector_features: set,
    ) -> _Cell:
        """
        Narrow a cell by the observed type and the design level.

        Args:
            cell (_Cell):
                The unrestricted cell.
            table_occupancy (str):
                Occupancy class as the tables key it.
            band (str):
                Story band the cell came from, empty for RES1.
            era (str):
                Construction era, empty outside the West Coast.
            design_level (str):
                The asset's design level.
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors, where a
                list is one value and not one value per world.

        Returns:
            _Cell:
                What is left to draw from, or a diagnosis naming the
                restriction that emptied the cell and the features that
                disagree.
        """
        candidates = list(zip(cell.types, cell.weights))
        general_type = ''
        if self.use_building_type:
            value = features.get(self.building_type_key)
            if value is not None:
                if is_probabilistic_feature(
                    self.building_type_key, value, vector_features
                ):
                    return _Cell(
                        f'{self.building_type_key} is a per-world list; '
                        'multi-world drawing is planned and not supported yet'
                    )
                general_type = str(value)
                members = self._membership.get(general_type)
                if members is None:
                    return _Cell(
                        f'{self.building_type_key} {general_type!r} is not a '
                        'Hazus general building type'
                    )
                kept = [
                    (structure_type, weight)
                    for structure_type, weight in candidates
                    if structure_type in members
                ]
                if not self._drawable(kept):
                    partner = self._blocking_partner(
                        table_occupancy, band, era, members
                    )
                    return _Cell(
                        f'{self.building_type_key} {general_type} has no '
                        'structure type in this cell',
                        RESTRICTION_BUILDING_TYPE,
                        f'{self.building_type_key}|{partner}'
                        if partner
                        else '',
                    )
                candidates = kept
        if design_level not in self._unreinforced_design_levels:
            kept = [
                (structure_type, weight)
                for structure_type, weight in candidates
                if structure_type not in UNREINFORCED_MASONRY_TYPES
            ]
            if not self._drawable(kept):
                # The design level is derived from the year, so the year is
                # what a caller can give up; the design level holds no rank
                # of its own.
                partner = (
                    self.building_type_key if general_type
                    else self.occupancy_key
                )
                return _Cell(
                    f'{self.design_level_key} {design_level} leaves no '
                    'structure type in this cell',
                    RESTRICTION_DESIGN_LEVEL,
                    f'{partner}|{self.year_built_key}',
                )
            candidates = kept
        return _Cell(
            DIAGNOSIS_OK,
            types=tuple(structure_type for structure_type, _ in candidates),
            weights=tuple(weight for _, weight in candidates),
        )

    def _blocking_partner(
        self, table_occupancy: str, band: str, era: str, members: list[str]
    ) -> str:
        """
        Name the feature that stands between a building type and a cell.

        The cell is selected by occupancy, story band and, on the West
        Coast, era. Each is tried in turn with the others held where the
        asset has them, so the feature named is one whose change on its
        own would admit the observed general type. Where no value of any
        of them would, there is no pair to name.

        Args:
            table_occupancy (str):
                Occupancy class as the tables key it.
            band (str):
                Story band the cell came from, empty for RES1.
            era (str):
                Construction era, empty outside the West Coast.
            members (list[str]):
                The Hazus types the observed general type allows.

        Returns:
            str:
                The feature key that pairs with the building type, or an
                empty string where changing none of them would help.
        """
        if not band:
            # RES1 is tabulated without a story band, so a change of
            # occupancy carries the asset into whichever band its story
            # count gives it, and any band that admits the type counts.
            if any(
                self._admits(other_band, row, members)
                for other_occupancy in self._weights
                if other_occupancy != 'RES1'
                for other_band, row in self._occupancy_rows(other_occupancy)
            ):
                return self.occupancy_key
            return ''
        if self._needs_era and any(
            self._admits(band, self._row(table_occupancy, band, other), members)
            for other in self._eras()
            if other != era
        ):
            return self.year_built_key
        if any(
            self._admits(other, self._row(table_occupancy, other, era), members)
            for other in self._band_names()
            if other != band
        ):
            return self.stories_key
        if any(
            self._admits(band, self._row(other, band, era), members)
            for other in self._weights
            if other not in (table_occupancy, 'RES1')
        ):
            return self.occupancy_key
        return ''

    def _eras(self) -> tuple:
        """
        Return the construction eras the region's tables carry.

        Returns:
            tuple:
                Era names on the West Coast, a single empty name
                elsewhere.
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
        return tuple(band for band in self._type_lists if band != 'RES1')

    def _row(self, table_occupancy: str, band: str, era: str) -> list:
        """
        Return one weight row by occupancy, band, and era.

        Args:
            table_occupancy (str):
                Occupancy class as the tables key it.
            band (str):
                Story band.
            era (str):
                Construction era, empty outside the West Coast.

        Returns:
            list:
                The weights of that cell.
        """
        cell = self._weights[table_occupancy][band]
        return cell[era] if self._needs_era else cell

    def _occupancy_rows(self, table_occupancy: str) -> list:
        """
        Return every band and weight row of one occupancy.

        Args:
            table_occupancy (str):
                Occupancy class as the tables key it.

        Returns:
            list:
                Pairs of the band the row belongs to and the row.
        """
        return [
            (band, self._row(table_occupancy, band, era))
            for band in self._weights[table_occupancy]
            for era in self._eras()
        ]

    def _admits(self, band: str, weights: list, members: list[str]) -> bool:
        """
        Say whether a weight row carries any of a general type's members.

        Args:
            band (str):
                Band whose type list the row aligns with.
            weights (list):
                The row.
            members (list[str]):
                The Hazus types the observed general type allows.

        Returns:
            bool:
                True where at least one member carries weight.
        """
        if not weights:
            return False
        return any(
            weight > 0
            for structure_type, weight in zip(self._type_lists[band], weights)
            if structure_type in members
        )

    @staticmethod
    def _drawable(candidates: list[tuple[str, float]]) -> bool:
        """
        Say whether a restricted cell still holds a drawable type.

        Args:
            candidates (list[tuple[str, float]]):
                Structure types and their weights.

        Returns:
            bool:
                True where at least one type carries weight.
        """
        return bool(candidates) and sum(
            weight for _, weight in candidates
        ) > 0
