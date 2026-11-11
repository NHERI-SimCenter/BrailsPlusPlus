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
Re-draw the NSI members of implausible occupancy and story count pairs.

The section "Pair re-draw" of the README beside this module states the
action table and the procedure.

.. autosummary::

      PairRedraw
      PairRedrawResult
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import pandas as pd

from brails.aggregators.points_to_polygons.nsi_footprint_join import (
    UNIT_CLASSES,
    point_class,
    unit_contribution,
)
from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    get_hazus_sampling_height_band,
    get_hazus_story_count,
)
from brails.inferers.hazus_inferer_earthquake.hazus_table_completion import (
    is_implausible,
)
from brails.types.asset_inventory import (
    HAZUS_EQ_SOURCE_FEATURES,
    SOURCE_CODES,
    SOURCE_STRING_FEATURE,
    AssetInventory,
)
from brails.utils.deterministic_sampling import DEFAULT_WORLD
from brails.utils.source_updates import FIXED_SOURCES, WITHHELD_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
        DonorDraw,
        DonorPool,
        ObservedDonorImputer,
    )
    from brails.inferers.stories_from_height_inferer.stories_from_height_inferer import (
        StoriesFromHeightInferer,
    )

# Prefix of the purpose labels of the re-draw's donor picks.
PURPOSE_PAIR_REDRAW = 'pair_redraw'

ACTION_REDRAW_OCCUPANCY = 'redraw_occupancy'
ACTION_REDRAW_PAIR = 'redraw_pair'
ACTION_REPLACE_STORIES = 'replace_stories'
ACTION_REPLACE_STORIES_REDRAW_OCCUPANCY = 'replace_stories_redraw_occupancy'
ACTION_LOG_LOCAL = 'log_local'
ACTION_LOG_IMPUTED = 'log_imputed'
REDRAW_ACTIONS = (
    ACTION_REDRAW_OCCUPANCY,
    ACTION_REDRAW_PAIR,
    ACTION_REPLACE_STORIES_REDRAW_OCCUPANCY,
)
REPLACE_ACTIONS = (ACTION_REPLACE_STORIES, ACTION_REPLACE_STORIES_REDRAW_OCCUPANCY)

# Outcome logged for an asset that takes no draw.
OUTCOME_LOGGED = 'logged'

PLAN_COLUMNS = (
    'asset_id',
    'occupancy',
    'stories',
    'band',
    'occupancy_code',
    'stories_code',
    'height_stories',
    'corroborated',
    'action',
)
LOG_COLUMNS = (
    *PLAN_COLUMNS,
    'world',
    'outcome',
    'attempts',
    'iteration',
    'donor_id',
)

_LOCAL_CODES = tuple(SOURCE_CODES[label] for label in FIXED_SOURCES)
_NSI_CODE = SOURCE_CODES['nsi']
_IMPUTED_CODE = SOURCE_CODES['imputed']
_DERIVED_CODE = SOURCE_CODES['derived']
# The source codes the action table covers: README.rst beside this module,
# "The action table".
_CLASS_CODES = (_NSI_CODE, *_LOCAL_CODES, _IMPUTED_CODE)
_STORY_CODES = (_NSI_CODE, _DERIVED_CODE, *_LOCAL_CODES, _IMPUTED_CODE)
_LABEL_OF_CODE = {code: label for label, code in SOURCE_CODES.items()}


class PairRedrawResult(NamedTuple):
    """
    The outcome of one world's pair re-draw.

    Attributes:
        updates (dict[Hashable, dict[str, object]]):
            The new values per asset id, for the assets whose re-draw found
            a donor, in ascending order of the id as a string.
        withheld (pandas.DataFrame):
            One row per replaced value, the replaced story counts included,
            with columns
            :data:`~brails.utils.source_updates.WITHHELD_COLUMNS`. The
            caller writes ``updates`` and ``withheld`` into the per-source
            inventories with
            :func:`~brails.utils.source_updates.apply_to_sources`.
        log (pandas.DataFrame):
            One row per asset with an implausible pair, with columns
            :data:`LOG_COLUMNS`.
        derived (dict[Hashable, dict[str, object]]):
            The height counts that replace story counts, per asset id in
            ascending order of the id as a string. The caller then writes
            them with the same function, passing them as the updates, an
            empty withheld table and ``imputed_label='derived'``.
    """

    updates: dict[Hashable, dict[str, object]]
    withheld: pd.DataFrame
    log: pd.DataFrame
    derived: dict[Hashable, dict[str, object]]


class PairRedraw:
    """
    Re-draw the implausible pairs of one world's merged inventory.

    :meth:`plan` lists every asset with an implausible pair and the action
    its sources call for, and :meth:`redraw` returns the replaced and
    re-drawn values without changing the inventory. Local and footprint
    values and derived story counts never change, and a pair with a local,
    footprint or imputed class, or an imputed story count, is only logged.
    Of a pair with an NSI class, an NSI story count is replaced by the
    count the footprint height gives where that count falls in another
    story band, and the class is then re-drawn from one observed donor if
    the pair stays implausible. Without a height count the class and the
    NSI count are re-drawn together, and otherwise the class is re-drawn.
    The README beside this module states the action table and the
    procedure.
    """

    def __init__(
        self,
        imputer: ObservedDonorImputer,
        donors: AssetInventory,
        stories_inferer: StoriesFromHeightInferer,
        *,
        occupancy_key: str = 'OccupancyClass',
        stories_key: str = 'NumberOfStories',
        units_key: str = 'NumberOfUnits',
        source_string_key: str = SOURCE_STRING_FEATURE,
        source_features: Sequence[str] = HAZUS_EQ_SOURCE_FEATURES,
        world: int = DEFAULT_WORLD,
    ) -> None:
        """
        Configure the re-draw.

        Args:
            imputer (ObservedDonorImputer):
                The imputer whose :meth:`~ObservedDonorImputer.draw` supplies
                the donors. Its status must be the default,
                ``occupancy_key`` among :data:`UNIT_CLASSES`.
            donors (AssetInventory):
                The observed donors of the world, with one possible world.
            stories_inferer (StoriesFromHeightInferer):
                The inferer whose
                :meth:`~StoriesFromHeightInferer.stories_of` gives the
                count an asset's footprint height implies.
            occupancy_key (str, optional):
                The occupancy class feature. Defaults to ``'OccupancyClass'``.
            stories_key (str, optional):
                The story count feature. Defaults to ``'NumberOfStories'``.
            units_key (str, optional):
                The unit count feature. Defaults to ``'NumberOfUnits'``.
            source_string_key (str, optional):
                The feature holding the source string. Defaults to
                :data:`SOURCE_STRING_FEATURE`.
            source_features (Sequence[str], optional):
                The features the source string covers, in its order.
                Defaults to :data:`HAZUS_EQ_SOURCE_FEATURES`.
            world (int, optional):
                The world the inventory realizes. Defaults to
                :data:`DEFAULT_WORLD`.

        Raises:
            TypeError:
                If ``source_features`` is a single string.
            ValueError:
                If ``source_features`` repeats a feature or lacks the
                occupancy or story count feature, or the imputer reads
                another status.
        """
        if isinstance(source_features, str):
            msg = 'source_features must be a sequence of feature names.'
            raise TypeError(msg)
        source_features = tuple(source_features)
        if len(set(source_features)) != len(source_features):
            msg = 'source_features repeats a feature.'
            raise ValueError(msg)
        missing = [
            key for key in (occupancy_key, stories_key) if key not in source_features
        ]
        if missing:
            msg = f'source_features lacks {missing}.'
            raise ValueError(msg)
        status = (imputer.status_key, set(imputer.status_values))
        if status != (occupancy_key, set(UNIT_CLASSES)):
            msg = (
                'The re-draw passes residential statuses and needs an imputer '
                f'whose status is {occupancy_key} among {UNIT_CLASSES}.'
            )
            raise ValueError(msg)
        self.imputer = imputer
        self.donors = donors
        self.stories_inferer = stories_inferer
        self.occupancy_key = occupancy_key
        self.stories_key = stories_key
        self.units_key = units_key
        self.source_string_key = source_string_key
        self.source_features = source_features
        self.world = world

    def plan(self, inventory: AssetInventory) -> pd.DataFrame:
        """
        List the assets with an implausible pair and the action for each.

        The plan changes nothing. An asset whose class is not a string, or
        whose story count cannot be read, is not listed.

        Args:
            inventory (AssetInventory):
                A merged inventory of one possible world whose every asset
                carries the source string.

        Returns:
            pandas.DataFrame:
                One row per implausible asset with columns
                :data:`PLAN_COLUMNS`, in ascending order of the asset id
                as a string, holding the values as read (object dtype).
                ``stories`` is the count read, and ``height_stories`` the
                count the footprint height gives, or None.

        Raises:
            ValueError:
                If the inventory holds more than one possible world, an
                asset's source string is missing or of the wrong length, or
                an implausible pair's source codes fit no row of the action
                table.
        """
        if inventory.n_pw != 1:
            msg = (
                f'The inventory holds {inventory.n_pw} possible worlds. The '
                'pair re-draw reads one world at a time.'
            )
            raise ValueError(msg)
        rows = []
        for asset_id in sorted(inventory.inventory, key=str):
            features = inventory.inventory[asset_id].features
            codes = self._codes(asset_id, features)
            row = self._plan_row(asset_id, features, codes)
            if row is not None:
                rows.append(row)
        return pd.DataFrame(rows, columns=list(PLAN_COLUMNS), dtype=object)

    def redraw(self, inventory: AssetInventory) -> PairRedrawResult:
        """
        Replace and re-draw the planned pairs.

        Args:
            inventory (AssetInventory):
                A merged inventory of one possible world whose every asset
                carries the source string. It is not modified.

        Returns:
            PairRedrawResult:
                The new values, the withheld values, the log and the
                height counts.

        Raises:
            ValueError:
                As :meth:`plan` raises.
            RuntimeError:
                If a draw returns a donor whose values lack a re-drawn
                feature, which its test excludes.
        """
        plan = self.plan(inventory)
        pool = None
        updates: dict[Hashable, dict[str, object]] = {}
        derived: dict[Hashable, dict[str, object]] = {}
        withheld = []
        log = []
        for row in plan.to_dict('records'):
            asset_id = row['asset_id']
            features = inventory.inventory[asset_id].features
            codes = self._codes(asset_id, features)
            entry = {
                **row,
                'world': self.world,
                'outcome': OUTCOME_LOGGED,
                'attempts': 0,
                'iteration': None,
                'donor_id': None,
            }
            replaced = []
            kept_stories = row['stories']
            if row['action'] in REPLACE_ACTIONS:
                kept_stories = row['height_stories']
                derived[asset_id] = {self.stories_key: kept_stories}
                replaced.append(self.stories_key)
            if row['action'] in REDRAW_ACTIONS:
                drawn_features = (
                    (self.occupancy_key, self.stories_key)
                    if row['action'] == ACTION_REDRAW_PAIR
                    else (self.occupancy_key,)
                )
                if pool is None:
                    pool = self.imputer.pool(self.donors, inventory)
                draw = self._draw(
                    pool, asset_id, features, codes, row, drawn_features, kept_stories
                )
                entry.update(
                    outcome=draw.outcome,
                    attempts=draw.attempts,
                    iteration=draw.iteration,
                    donor_id=draw.donor_id,
                )
                if draw.donor_id is not None:
                    copied = dict(draw.values)
                    lacking = [f for f in drawn_features if f not in copied]
                    if lacking:
                        msg = (
                            f'The draw for asset {asset_id!r} returned donor '
                            f'{draw.donor_id!r} without {lacking}.'
                        )
                        raise RuntimeError(msg)
                    updates[asset_id] = {f: copied[f] for f in drawn_features}
                    replaced = [*drawn_features, *replaced]
            withheld.extend(
                {
                    'asset_id': asset_id,
                    'feature': feature,
                    'source': _LABEL_OF_CODE[
                        codes[self.source_features.index(feature)]
                    ],
                    'value': features[feature],
                }
                for feature in dict.fromkeys(replaced)
            )
            log.append(entry)
        return PairRedrawResult(
            updates,
            pd.DataFrame(withheld, columns=list(WITHHELD_COLUMNS), dtype=object),
            pd.DataFrame(log, columns=list(LOG_COLUMNS), dtype=object),
            derived,
        )

    def _codes(self, asset_id: Hashable, features: Mapping[str, Any]) -> str:
        """
        Return an asset's source string.

        Args:
            asset_id (Hashable):
                The asset.
            features (Mapping[str, Any]):
                Its features.

        Returns:
            str:
                The source string, one code per source feature.

        Raises:
            ValueError:
                If the string is missing or its length differs from the
                number of source features.
        """
        codes = features.get(self.source_string_key)
        if not isinstance(codes, str) or len(codes) != len(self.source_features):
            msg = (
                f'Asset {asset_id!r} carries no source string of '
                f'{len(self.source_features)} codes in '
                f'{self.source_string_key!r}: {codes!r}.'
            )
            raise ValueError(msg)
        return codes

    def _plan_row(
        self, asset_id: Hashable, features: Mapping[str, Any], codes: str
    ) -> dict[str, Any] | None:
        """
        Build an asset's plan row, if its pair is implausible.

        Args:
            asset_id (Hashable):
                The asset.
            features (Mapping[str, Any]):
                Its features.
            codes (str):
                Its source string.

        Returns:
            dict[str, Any] | None:
                The row keyed by :data:`PLAN_COLUMNS`, or None where the
                asset forms no implausible pair.

        Raises:
            ValueError:
                If the pair's source codes fit no row of the action table.
        """
        occupancy = features.get(self.occupancy_key)
        stories = _story_count(features.get(self.stories_key))
        if not isinstance(occupancy, str) or stories is None:
            return None
        if not is_implausible(occupancy, stories):
            return None
        occupancy_code = codes[self.source_features.index(self.occupancy_key)]
        stories_code = codes[self.source_features.index(self.stories_key)]
        band = get_hazus_sampling_height_band(stories)
        height_stories = self.stories_inferer.stories_of(features)
        corroborated = (
            height_stories is not None
            and get_hazus_sampling_height_band(height_stories) == band
        )
        return {
            'asset_id': asset_id,
            'occupancy': occupancy,
            'stories': stories,
            'band': band,
            'occupancy_code': occupancy_code,
            'stories_code': stories_code,
            'height_stories': height_stories,
            'corroborated': corroborated,
            'action': _action(
                asset_id,
                occupancy_code,
                stories_code,
                corroborated=corroborated,
                height_pair_implausible=(
                    None
                    if height_stories is None
                    else is_implausible(occupancy, height_stories)
                ),
            ),
        }

    def _draw(
        self,
        pool: DonorPool,
        asset_id: Hashable,
        features: Mapping[str, Any],
        codes: str,
        row: Mapping[str, Any],
        drawn_features: tuple[str, ...],
        kept_stories: int,
    ) -> DonorDraw:
        """
        Call the imputer's draw for one planned asset.

        The draw is given the asset's residential status and the test
        :meth:`_admit` builds, and may not change its local and footprint
        features.

        Args:
            pool (DonorPool):
                The pool of the donors and the inventory.
            asset_id (Hashable):
                The asset.
            features (Mapping[str, Any]):
                Its features in this world.
            codes (str):
                Its source string.
            row (Mapping[str, Any]):
                Its plan row.
            drawn_features (tuple[str, ...]):
                The features re-drawn.
            kept_stories (int):
                The story count the class must pair with when the count is
                not re-drawn: the asset's, or the height count replacing it.

        Returns:
            DonorDraw:
                The imputer's draw.
        """
        residential = row['occupancy'] in UNIT_CLASSES
        protected = tuple(
            feature
            for feature, code in zip(self.source_features, codes)
            if code in _LOCAL_CODES
        )
        admit = self._admit(
            drawn_features,
            kept_stories=kept_stories,
            residential=residential,
            units=features.get(self.units_key),
        )
        return self.imputer.draw(
            pool,
            asset_id,
            dict(features),
            drawn_features,
            world=self.world,
            iteration=0,
            admit=admit,
            status=residential,
            protected=protected,
            scan_nearest=True,
            purpose_prefix=PURPOSE_PAIR_REDRAW,
        )

    def _admit(
        self,
        drawn_features: tuple[str, ...],
        *,
        kept_stories: int,
        residential: bool,
        units: object,
    ) -> Callable[[Mapping[str, Any]], bool]:
        """
        Build the test a candidate donor's values must pass.

        The README beside this module, "The procedure", states the test.

        Args:
            drawn_features (tuple[str, ...]):
                The features re-drawn.
            kept_stories (int):
                The story count kept when ``drawn_features`` does not hold
                the story count feature.
            residential (bool):
                The asset's residential status.
            units (object):
                The asset's unit count, or a missing value.

        Returns:
            Callable[[Mapping[str, Any]], bool]:
                The test.
        """

        def admit(values: Mapping[str, Any]) -> bool:
            """
            Judge one candidate's values.

            Args:
                values (Mapping[str, Any]):
                    The candidate's values for the re-drawn features.

            Returns:
                bool:
                    Whether the candidate passes the test.
            """
            if any(feature not in values for feature in drawn_features):
                return False
            occupancy = values[self.occupancy_key]
            stories = (
                _story_count(values[self.stories_key])
                if self.stories_key in drawn_features
                else kept_stories
            )
            if not isinstance(occupancy, str) or stories is None:
                return False
            if (occupancy in UNIT_CLASSES) != residential:
                return False
            if is_implausible(occupancy, stories):
                return False
            if not residential:
                return True
            counted, _ = unit_contribution(occupancy, units)
            return point_class(occupancy, counted) == occupancy

        return admit


def _action(
    asset_id: Hashable,
    occupancy_code: str,
    stories_code: str,
    *,
    corroborated: bool,
    height_pair_implausible: bool | None,
) -> str:
    """
    Select the action the README's action table gives an implausible pair.

    Args:
        asset_id (Hashable):
            The asset, for the error message.
        occupancy_code (str):
            The source code of the occupancy class.
        stories_code (str):
            The source code of the story count.
        corroborated (bool):
            Whether the footprint height corroborates the story count.
        height_pair_implausible (bool | None):
            Whether the class and the count the footprint height gives form
            an implausible pair, None when the height gives no count.

    Returns:
        str:
            One of :data:`ACTION_REDRAW_OCCUPANCY`,
            :data:`ACTION_REDRAW_PAIR`, :data:`ACTION_REPLACE_STORIES`,
            :data:`ACTION_REPLACE_STORIES_REDRAW_OCCUPANCY`,
            :data:`ACTION_LOG_LOCAL` and :data:`ACTION_LOG_IMPUTED`.

    Raises:
        ValueError:
            If either code is outside the table, as a class coded ``R``,
            ``D`` or ``-``, or a story count coded ``D`` or ``-``.
    """
    if occupancy_code not in _CLASS_CODES or stories_code not in _STORY_CODES:
        msg = (
            f'Asset {asset_id!r} has an implausible pair with class code '
            f'{occupancy_code!r} and story count code {stories_code!r}, which '
            'the action table in brails/compilers/README.rst does not cover.'
        )
        raise ValueError(msg)
    if _IMPUTED_CODE in (occupancy_code, stories_code):
        return ACTION_LOG_IMPUTED
    if occupancy_code in _LOCAL_CODES:
        return ACTION_LOG_LOCAL
    if stories_code in (*_LOCAL_CODES, _DERIVED_CODE) or corroborated:
        return ACTION_REDRAW_OCCUPANCY
    if height_pair_implausible is None:
        return ACTION_REDRAW_PAIR
    if height_pair_implausible:
        return ACTION_REPLACE_STORIES_REDRAW_OCCUPANCY
    return ACTION_REPLACE_STORIES


def _story_count(value: object) -> int | None:
    """
    Read a story count as the structure type inferer reads it.

    Args:
        value (object):
            A feature value.

    Returns:
        int | None:
            The count
            :func:`~brails.inferers.hazus_inferer_earthquake.hazus_rulesets.get_hazus_story_count`
            reads, or None where it refuses the value.
    """
    try:
        return get_hazus_story_count(value)
    except (TypeError, ValueError):
        return None
