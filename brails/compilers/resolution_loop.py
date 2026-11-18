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


"""
Screen assets for the Hazus earthquake chain, withholding NSI types to admit them.

.. autosummary::

      ComposedScreen
      ResolutionLoop
      ResolutionResult
      ScreenVerdict
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from brails.constants import BLOCK_GROUP_KEY
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
)
from brails.types.asset_inventory import (
    SOURCE_CODES,
    Asset,
    AssetInventory,
    is_missing_value,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from brails.compilers.compilation_report import CompilationReport
    from brails.inferers.hazus_inferer_earthquake.hazus_eq_design_level_inferer import (
        HazusEQDesignLevelInferer,
    )
    from brails.inferers.hazus_inferer_earthquake.hazus_eq_structure_type_inferer import (
        HazusEQStructureTypeInferer,
    )

# The stages of the composed screen, in the order it runs them. Each name
# is also a stage of the compilation report.
STAGE_DESIGN_LEVEL = 'design_level'
STAGE_STRUCTURE_TYPE = 'structure_type'
SCREEN_STAGES = (STAGE_DESIGN_LEVEL, STAGE_STRUCTURE_TYPE)

# The asset id the screen files one asset's features under. A diagnosis
# is a function of an asset's features, so any id serves.
SCREEN_ASSET_ID = 'screened_asset'

# The coordinate every asset the screen builds carries. An asset requires
# a valid coordinate and no stage of the screen reads one.
SCREEN_COORDINATES = ((0.0, 0.0),)

# The stage of the compilation report the loop records under.
STAGE_RESOLUTION = 'resolution'

# The source whose general building type the loop withholds.
WITHHELD_SOURCE = 'nsi'


class ScreenVerdict(NamedTuple):
    """
    What the composed screen decided about one asset.

    Attributes:
        diagnosis (str):
            The verdict, from the shared diagnosis vocabulary.
        detail (str):
            What happened, in words.
        stage (str):
            The stage that refused the asset, one of
            :data:`SCREEN_STAGES`. Empty where the asset passed.
        candidates (tuple[str, ...]):
            The features the refusing stage names, each of which, changed
            alone, would admit the asset, in the order the stage named
            them. A named design level appears as the construction year it
            is derived from. Empty where the asset passed and where the
            stage named none.
    """

    diagnosis: str
    detail: str = ''
    stage: str = ''
    candidates: tuple[str, ...] = ()


class ResolutionResult(NamedTuple):
    """
    The assets a resolution run admitted, and the ones it excluded.

    Attributes:
        resolved (AssetInventory):
            The admitted assets, with copies of the features they pass the
            screen with, whose values are shared with the input inventory.
            A withheld building type is absent. Where the input carries the design level, the
            structure type draw reads this inventory as it is.
        excluded (list):
            The ids of the excluded assets, in the order of the input's
            ids.
        given_up (dict):
            Asset id to the features withheld from its screen, for the
            admitted assets that gave up any.
        passed_first_screen (int):
            Assets admitted with nothing withheld.
        resolved_by_withholding (int):
            Assets admitted once their building type was withheld.
        excluded_count (int):
            The length of ``excluded``.
    """

    resolved: AssetInventory
    excluded: list
    given_up: dict
    passed_first_screen: int
    resolved_by_withholding: int
    excluded_count: int


def _one_asset_inventory(
    features: Mapping[str, object],
    vector_features: frozenset[str] | Sequence[str],
) -> AssetInventory:
    """
    Wrap one asset's features in an inventory a component can read.

    Args:
        features (Mapping[str, object]):
            The asset's features. Copied, so the component cannot reach
            the caller's mapping.
        vector_features (frozenset[str] or Sequence[str]):
            Names of the features whose values are vectors.

    Returns:
        AssetInventory:
            An inventory holding that one asset under
            :data:`SCREEN_ASSET_ID`.
    """
    inventory = AssetInventory(vector_features=list(vector_features))
    inventory.add_asset(
        SCREEN_ASSET_ID,
        Asset(
            SCREEN_ASSET_ID,
            [list(point) for point in SCREEN_COORDINATES],
            dict(features),
        ),
    )
    return inventory


def _add_design_levels(
    inferer: HazusEQDesignLevelInferer, inventory: AssetInventory
) -> None:
    """
    Give each asset without a design level the level its year implies.

    Args:
        inferer (HazusEQDesignLevelInferer):
            The inferer that derives the level from the construction
            year. It must take every asset of the inventory, so each year
            is one its table reads.
        inventory (AssetInventory):
            The inventory whose assets receive the levels, changed in
            place. Both callers pass an inventory of copies they built, so
            the assets the screen was given are not changed.
    """
    key = inferer.design_level_key
    missing = [
        asset_id
        for asset_id in inventory.get_asset_ids()
        if inventory.get_asset_features(asset_id)[1].get(key) is None
    ]
    if missing:
        levels = inferer.draw(inventory)
        for asset_id in missing:
            inventory.add_asset_features(asset_id, {key: levels.loc[asset_id]})


def _category(
    screen: ComposedScreen,
    verdict: ScreenVerdict,
    features: Mapping[str, object],
    *,
    acted_on: bool,
    vector_features: frozenset[str] | Sequence[str] = frozenset(),
) -> str:
    """
    Settle which exclusion category a refused asset falls in.

    The section "The exclusion category" of the README beside this module
    states the rule.

    Args:
        screen (ComposedScreen):
            The screen that refused the asset.
        verdict (ScreenVerdict):
            The screen's refusal of the asset.
        features (Mapping[str, object]):
            The asset's features, as the screen read them.
        acted_on (bool):
            Whether a caller acted on a candidate of the refusal and the
            asset failed anyway.
        vector_features (frozenset[str] or Sequence[str]):
            Names of the features whose values are vectors.

    Returns:
        str:
            A category of
            :data:`~brails.compilers.compilation_report.EXCLUSION_CATEGORIES`.
    """
    if verdict.diagnosis == DIAGNOSIS_UNSUPPORTED:
        return 'unsupported'
    if verdict.diagnosis == DIAGNOSIS_INVALID:
        return 'missing_driver'
    if acted_on:
        return 'conflict'
    if verdict.stage != STAGE_STRUCTURE_TYPE:
        return 'undetermined'
    inventory = _one_asset_inventory(features, vector_features)
    _add_design_levels(screen.design_level_inferer, inventory)
    _, cell = inventory.get_asset_features(SCREEN_ASSET_ID)
    inferer = screen.structure_type_inferer
    vector = frozenset(vector_features)
    filled = inferer.cell_fill(cell, vector_features=vector) is not None
    if not filled and inferer.cell_is_unpublished(cell, vector_features=vector):
        return 'empty_cell'
    return 'conflict'


class ComposedScreen:
    """
    Ask the Hazus earthquake inference chain whether it can take an asset.

    The design level inferer and then the structure type inferer check the
    asset, and the verdict is the first refusal, or a pass. The screen never
    changes the features it is given, since it writes design levels on
    copies of the assets. It samples no construction year, so an asset
    without one is refused at the design level stage. The screen reads an
    inventory of one possible world, and both inferers refuse as unsupported
    a feature that holds one value per world. The section "Composed screen"
    of the README beside this module states the procedure.
    """

    def __init__(
        self,
        design_level_inferer: HazusEQDesignLevelInferer,
        structure_type_inferer: HazusEQStructureTypeInferer,
    ) -> None:
        """
        Set up the screen with the components it runs.

        Args:
            design_level_inferer (HazusEQDesignLevelInferer):
                The first stage, which reads the construction year.
            structure_type_inferer (HazusEQStructureTypeInferer):
                The second stage, which reads the year and the level.
                Pass the inferer the structure type draw uses, so that
                the screen and the draw read one table.

        Raises:
            ValueError:
                If the two inferers read the construction year or the
                design level under different feature names.
        """
        for key in ('year_built_key', 'design_level_key'):
            first = getattr(design_level_inferer, key)
            second = getattr(structure_type_inferer, key)
            if first != second:
                msg = (
                    f'The design level inferer has {key} {first!r} and the '
                    f'structure type inferer {second!r}; the second stage '
                    'reads what the first one reads and writes.'
                )
                raise ValueError(msg)
        self.design_level_inferer = design_level_inferer
        self.structure_type_inferer = structure_type_inferer

    def screen(
        self,
        features: Mapping[str, object],
        *,
        vector_features: frozenset[str] | Sequence[str] = frozenset(),
    ) -> ScreenVerdict:
        """
        Run the chain over one asset's features.

        Args:
            features (Mapping[str, object]):
                One asset's features. Read and not written.
            vector_features (frozenset[str] or Sequence[str]):
                Names of the features whose values are vectors.

        Returns:
            ScreenVerdict:
                The verdict of the first stage that refused the asset, or
                a passing verdict naming no stage.
        """
        inventory = _one_asset_inventory(features, vector_features)
        return self.screen_inventory(inventory)[SCREEN_ASSET_ID]

    def screen_inventory(self, inventory: AssetInventory) -> dict:
        """
        Run the chain over every asset of an inventory.

        Each stage checks all the assets it receives in one call, and an
        asset gets the verdict :meth:`screen` gives it alone. Where an
        asset carries a construction year and no design level, the
        structure type stage reads the level the year implies.

        Args:
            inventory (AssetInventory):
                The assets to screen. Read and not written.

        Returns:
            dict:
                Asset id to its :class:`ScreenVerdict`, in the order of
                the inventory's ids.
        """
        first = self.design_level_inferer.check(inventory)
        verdicts = {}
        taken = AssetInventory(vector_features=list(inventory.vector_features))
        for asset_id in inventory.get_asset_ids():
            row = first.loc[asset_id]
            if row['diagnosis'] != DIAGNOSIS_OK:
                verdicts[asset_id] = self._refusal(row, STAGE_DESIGN_LEVEL)
                continue
            verdicts[asset_id] = ScreenVerdict(DIAGNOSIS_OK)
            _, features = inventory.get_asset_features(asset_id)
            taken.add_asset(
                asset_id,
                Asset(
                    asset_id,
                    [list(point) for point in SCREEN_COORDINATES],
                    dict(features),
                ),
            )
        _add_design_levels(self.design_level_inferer, taken)
        second = self.structure_type_inferer.check(taken)
        for asset_id in taken.get_asset_ids():
            row = second.loc[asset_id]
            if row['diagnosis'] != DIAGNOSIS_OK:
                verdicts[asset_id] = self._refusal(row, STAGE_STRUCTURE_TYPE)
        return verdicts

    def _refusal(self, row: pd.Series, stage: str) -> ScreenVerdict:
        """
        Build the verdict of a stage that refused an asset.

        Args:
            row (pd.Series):
                The asset's row of the stage's diagnosis frame.
            stage (str):
                The stage, one of :data:`SCREEN_STAGES`.

        Returns:
            ScreenVerdict:
                A verdict holding the stage, its diagnosis and detail, and
                the features it names, with the construction year in place
                of a named design level.
        """
        year_key = self.design_level_inferer.year_built_key
        level_key = self.design_level_inferer.design_level_key
        # A design level is derived from the year, so the year is the
        # feature to change. Neither HazusEQDesignLevelInferer nor
        # HazusEQStructureTypeInferer names the level. The mapping, and
        # dict.fromkeys dropping the repeat it can cause, cover a stage that
        # does.
        candidates = dict.fromkeys(
            year_key if feature == level_key else feature
            for feature in row['resolvable_by']
        )
        return ScreenVerdict(
            row['diagnosis'], row['detail'], stage, tuple(candidates)
        )


class ResolutionLoop:
    """
    Withhold the NSI building type of a refused asset and screen it again.

    The loop screens one world's merged inventory through
    :meth:`ComposedScreen.screen_inventory`. Where a refusal names the
    building type and :data:`WITHHELD_SOURCE` holds it, the type is
    withheld and the asset screened again, the loop's one action. An
    unsupported asset is excluded without action. Every other refused
    asset, and one still refused without its type, is excluded in the
    world of the screen's structure type inferer. The loop
    draws nothing and does not change the inventory it reads. The section
    "Resolution loop" of the README beside this module states what is
    recorded and how an exclusion is filed.
    """

    def __init__(
        self,
        screen: ComposedScreen,
        *,
        block_group_key: str = BLOCK_GROUP_KEY,
    ) -> None:
        """
        Set up the loop with the screen it tests against.

        The building type key and the world are those of the screen's
        structure type inferer.

        Args:
            screen (ComposedScreen):
                The screen every attempt is tested against.
            block_group_key (str):
                Feature name holding the asset's Census block group,
                recorded against every action.
        """
        self.screen = screen
        self.block_group_key = block_group_key

    def resolve(
        self,
        inventory: AssetInventory,
        report: CompilationReport,
        *,
        building_type_sources: Mapping[str | int, str],
    ) -> ResolutionResult:
        """
        Admit or exclude each asset of one world's inventory.

        ``building_type_sources`` is checked for every asset before the
        first is screened, so an input the loop refuses leaves the report
        as it was.

        Args:
            inventory (AssetInventory):
                The world's merged inventory of scalar values. Read and not
                written.
            report (CompilationReport):
                Where every action and exclusion is recorded.
            building_type_sources (Mapping[str | int, str]):
                Asset id to the label of the source holding the asset's
                general building type, a key of
                :data:`~brails.types.asset_inventory.SOURCE_CODES`, for
                every asset that holds one and no other.

        Returns:
            ResolutionResult:
                The admitted assets, the excluded ids, and how the assets
                were admitted.

        Raises:
            ValueError:
                If ``building_type_sources`` lacks an asset that holds a
                building type, gives one for an asset that holds none, or
                gives a label that is not a source.
        """
        key = self.screen.structure_type_inferer.building_type_key
        world = self.screen.structure_type_inferer.world
        for asset_id in inventory.get_asset_ids():
            _, features = inventory.get_asset_features(asset_id)
            type_source = building_type_sources.get(asset_id)
            holds_type = not is_missing_value(features.get(key))
            if (holds_type and type_source not in SOURCE_CODES) or (
                not holds_type and type_source is not None
            ):
                msg = (
                    f'building_type_sources gives asset {asset_id!r} '
                    f'{type_source!r}. An asset that holds a {key} needs the '
                    f'label of its source, one of {sorted(SOURCE_CODES)}, and '
                    'an asset that holds none needs no entry.'
                )
                raise ValueError(msg)

        vector_features = frozenset(inventory.vector_features)
        resolved = AssetInventory(vector_features=list(inventory.vector_features))
        excluded: list = []
        given_up: dict = {}
        for asset_id, verdict in self.screen.screen_inventory(inventory).items():
            _, features = inventory.get_asset_features(asset_id)
            admitted = dict(features) if verdict.diagnosis == DIAGNOSIS_OK else None
            # An unsupported asset is excluded without action, whatever its
            # refusal names.
            acted_on = (
                verdict.diagnosis not in (DIAGNOSIS_OK, DIAGNOSIS_UNSUPPORTED)
                and key in verdict.candidates
                and building_type_sources.get(asset_id) == WITHHELD_SOURCE
            )
            detail = verdict.detail
            if acted_on:
                report.record_action(
                    asset_id,
                    key,
                    stage=STAGE_RESOLUTION,
                    action='withheld',
                    world=world,
                    old=features[key],
                    source=WITHHELD_SOURCE,
                    reason=(
                        'the screen named it as a feature whose removal would '
                        'admit the asset'
                    ),
                    block_group=features.get(self.block_group_key),
                )
                kept = {
                    feature: value
                    for feature, value in features.items()
                    if feature != key
                }
                retry = self.screen.screen(kept, vector_features=vector_features)
                if retry.diagnosis == DIAGNOSIS_OK:
                    admitted = kept
                    given_up[asset_id] = (key,)
                else:
                    report.record_action(
                        asset_id,
                        key,
                        stage=STAGE_RESOLUTION,
                        action='restored',
                        world=world,
                        new=features[key],
                        source=WITHHELD_SOURCE,
                        reason='withholding it did not admit the asset',
                        block_group=features.get(self.block_group_key),
                    )
                    detail = f'{verdict.detail}; without {key}: {retry.detail}'
            if admitted is None:
                excluded.append(asset_id)
                report.record_exclusion(
                    asset_id,
                    world=world,
                    # An asset the loop took no action on was excluded by
                    # the stage that refused it.
                    stage=STAGE_RESOLUTION if acted_on else verdict.stage,
                    category=_category(
                        self.screen,
                        verdict,
                        features,
                        acted_on=acted_on,
                        vector_features=vector_features,
                    ),
                    diagnosis=verdict.diagnosis,
                    detail=detail,
                    resolvable_by=verdict.candidates,
                )
                continue
            _, coordinates = inventory.get_asset_coordinates(asset_id)
            resolved.add_asset(asset_id, Asset(asset_id, coordinates, admitted))

        admitted_count = len(resolved.get_asset_ids())
        return ResolutionResult(
            resolved=resolved,
            excluded=excluded,
            given_up=given_up,
            passed_first_screen=admitted_count - len(given_up),
            resolved_by_withholding=len(given_up),
            excluded_count=len(excluded),
        )
