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
Allocate a block group's census housing units to footprints without records.

.. autosummary::

      BlockGroupUnitBudget
      UnitBudgetResult
"""

from __future__ import annotations

import math
import secrets
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from brails.aggregators.points_to_polygons.nsi_footprint_join import (
    COUNT_BANDS,
    GROSS_AREA,
    UNIT_CLASSES,
    UNIT_MIDPOINTS,
    unit_contribution,
)
from brails.constants import BLOCK_GROUP_KEY, GEOGRAPHIC_CRS, GEOID_LENGTH
from brails.types.asset_inventory import Asset, AssetInventory, is_missing_value
from brails.utils.asset_id import EARTH_RADIUS_M
from brails.utils.deterministic_sampling import asset_key_digest, keyed_uniforms
from brails.utils.source_updates import WITHHELD_COLUMNS
from brails.utils.unit_converter import UNIT_TO_METER
from brails.utils.world_values import (
    OCCUPANCY_KEY,
    STORIES_KEY,
    UNITS_KEY,
    collapse_realizations,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
        ObservedDonorImputer,
    )

# Purpose label of the per-world order in which dwellings receive bins.
PURPOSE_UNIT_BUDGET = 'unit_budget'

# The B25024 structure-size bins, by label, with the table lines each sums.
# The table and the Hazus classes the bins take: README.rst beside this
# module, "The census table and the classes".
B25024_BINS = {
    '1': ('002', '003'),
    '2': ('004',),
    '3-4': ('005',),
    '5-9': ('006',),
    '10-19': ('007',),
    '20-49': ('008',),
    '50+': ('009',),
}

# The Hazus occupancy class of each bin. The unit ranges of the B25024 bins
# are the count bands of the Hazus classes (COUNT_BANDS), so a bin's range
# is its class's band.
BIN_CLASSES = {
    '1': 'RES1',
    '2': 'RES3A',
    '3-4': 'RES3B',
    '5-9': 'RES3C',
    '10-19': 'RES3D',
    '20-49': 'RES3E',
    '50+': 'RES3F',
}

# The B25024 line of mobile homes. Line 011, boat, RV, van, is not used.
MOBILE_HOME_BIN = '010'

# Labels of the mobile-home bin and of the sum of the structure-size bins
# in the diagnostic and the gaps.
MOBILE_HOME_LABEL = 'mobile_home'
TOTAL_LABEL = 'total'

# The class and unit count a mobile home receives.
MOBILE_HOME_CLASS = 'RES2'
MOBILE_HOME_UNITS = 1

# The class of a reached footprint the mobile-home rule may reclassify.
RELABEL_CLASS = 'RES1'

# The table name and the block-group column of the B25024 table.
B25024_TABLE = 'B25024'
GEOID_COLUMN = 'geoid'

# Under drop_without_acs, a small target in a block group without a budget
# is dropped when its residential probability is below this.
DROP_PROBABILITY = 0.5

# Outcomes of the log.
OUTCOME_MOBILE_HOME = 'mobile_home'
OUTCOME_RELABELED = 'relabeled'
OUTCOME_DWELLING = 'dwelling'
OUTCOME_DROPPED = 'dropped'
OUTCOME_NONRESIDENTIAL = 'nonresidential'
OUTCOME_NO_ACS = 'no_acs'

# Where a site's gross floor area comes from: its GROSS_AREA, its plan area
# times its STORIES_KEY, or its plan area alone, one story.
AREA_FROM_GROSS_AREA = 'gross_area'
AREA_FROM_STORIES = 'stories'
AREA_FROM_PLAN_AREA = 'plan_area'

# Kinds of budget gap. A candidate shortfall without a bin counts the
# structures beyond the number of candidates, and one with a bin the
# structures of that bin no remaining candidate can hold. Units clipped are
# those of a bin beyond its structures' range, and units over capacity
# those its dwellings' capacities cannot hold.
GAP_NO_ACS = 'no_acs'
GAP_MOBILE_HOME_SHORTFALL = 'mobile_home_shortfall'
GAP_CANDIDATE_SHORTFALL = 'candidate_shortfall'
GAP_STRUCTURES_WITHOUT_UNITS = 'structures_without_units'
GAP_BIN_WITHOUT_STRUCTURE = 'bin_without_structure'
GAP_UNITS_CLIPPED = 'units_clipped'
GAP_UNITS_OVER_CAPACITY = 'units_over_capacity'

LOG_COLUMNS = (
    'asset_id',
    'block_group',
    'reached',
    'outcome',
    'probability',
    'rank',
    'plan_area_sqft',
    'gross_area_source',
    'stories',
    'capacity',
    'mh_park_distance_m',
)
GAP_COLUMNS = ('block_group', 'bin', 'kind', 'count', 'units')
DIAGNOSTIC_COLUMNS = (
    'block_group',
    'world',
    'bin',
    'estimate',
    'margin',
    'reached_units',
    'generated_units',
    'total_units',
    'lower',
    'upper',
    'outside',
)


def _class_range(occupancy: str) -> tuple[int, int | None]:
    """
    Return the unit counts a dwelling of a housing class may hold.

    Args:
        occupancy (str):
            A class of
            :data:`~brails.aggregators.points_to_polygons.nsi_footprint_join.COUNT_BANDS`.

    Returns:
        tuple[int, int | None]:
            The least count, at least one, and the greatest, ``None`` for
            the open top band.
    """
    bounds = {band: lower for lower, band in COUNT_BANDS}
    above = [lower for lower, _ in COUNT_BANDS if lower > bounds[occupancy]]
    return max(1, bounds[occupancy]), (min(above) - 1 if above else None)


class UnitBudgetResult(NamedTuple):
    """
    The result of :meth:`BlockGroupUnitBudget.allocate`.

    Attributes:
        imputed (AssetInventory):
            :data:`~brails.utils.world_values.OCCUPANCY_KEY` and
            :data:`~brails.utils.world_values.UNITS_KEY` of each dwelling
            and mobile home, and the class alone of each reached footprint
            made a mobile home, with the allocation's number of possible
            worlds, stored by
            :func:`~brails.utils.world_values.collapse_realizations`.
        withheld (pandas.DataFrame):
            The class of each reached footprint made a mobile home, which
            the caller removes from its source before merging, with the
            columns of :data:`~brails.utils.source_updates.WITHHELD_COLUMNS`,
            as :func:`~brails.utils.source_updates.apply_to_sources` takes
            them, empty when ``allocate`` names no ``relabel_sources``.
        residential (dict[Hashable, bool]):
            The residential status of each target the budget decided: true
            for dwellings and mobile homes, false for the targets passed to
            the non-residential stage.
        dropped (tuple):
            The targets dropped from the inventory, in ascending order of
            the id as a string.
        nonresidential (tuple):
            The targets passed to the non-residential stage, in the same
            order.
        diagnostic (pandas.DataFrame):
            One row per block group with a budget, world and bin, the
            mobile-home bin and :data:`TOTAL_LABEL` included, with the
            columns of :data:`DIAGNOSTIC_COLUMNS`. ``generated_units`` holds
            the units the budget placed in that world, ``total_units`` those
            and the reached units, ``lower`` and ``upper`` the estimate less
            and plus :attr:`BlockGroupUnitBudget.moe_factor` times the
            margin, and ``outside`` whether the total lies beyond them. The
            margin of a bin summing two table lines, and of the total, is
            the root of the summed squared margins.
        gaps (pandas.DataFrame):
            The parts of a budget left unplaced, and the block groups
            without one, with the columns of :data:`GAP_COLUMNS`. ``count``
            holds structures, or targets for :data:`GAP_NO_ACS`. ``units``
            holds units, for :data:`GAP_UNITS_OVER_CAPACITY` the most of
            the bin's units left over in any one world.
        log (pandas.DataFrame):
            One row per target and per reached footprint made a mobile
            home, with the columns of :data:`LOG_COLUMNS`.
            ``gross_area_source`` says where the gross floor area comes
            from (:data:`AREA_FROM_GROSS_AREA`, :data:`AREA_FROM_STORIES`
            or :data:`AREA_FROM_PLAN_AREA`), ``stories`` is the story count
            that multiplied the plan area, NaN where none did, and
            ``capacity`` the units the gross floor area holds.
        seed (int):
            The seed of the per-world order.
    """

    imputed: AssetInventory
    withheld: pd.DataFrame
    residential: dict
    dropped: tuple
    nonresidential: tuple
    diagnostic: pd.DataFrame
    gaps: pd.DataFrame
    log: pd.DataFrame
    seed: int


class _BudgetRow(NamedTuple):
    """
    The B25024 counts of one block group.

    Attributes:
        estimate (dict[str, float]):
            The estimate per bin label and :data:`MOBILE_HOME_LABEL`.
        margin (dict[str, float]):
            The margin of error per label, zero where negative or missing.
    """

    estimate: dict
    margin: dict


class _BudgetSite(NamedTuple):
    """
    Where an asset is and how large it is.

    Attributes:
        area_m2 (float):
            Plan area in square meters, zero without area.
        block_group (str | None):
            The block-group GEOID.
        mh_park_m (float):
            Distance to the nearest mobile-home park point in meters,
            infinite without area or mobile-home park points.
        gross_m2 (float):
            The gross floor area in square meters.
        capacity (int):
            The whole number of units the gross floor area holds.
        gross_source (str):
            Where the gross floor area comes from:
            :data:`AREA_FROM_GROSS_AREA`, :data:`AREA_FROM_STORIES` or
            :data:`AREA_FROM_PLAN_AREA`.
        stories (float):
            The story count that multiplied the plan area, NaN where none
            did.
    """

    area_m2: float
    block_group: str | None
    mh_park_m: float
    gross_m2: float
    capacity: int
    gross_source: str
    stories: float


class _BudgetGroup(NamedTuple):
    """
    One block group with its counts and footprints.

    Attributes:
        geoid (str | None):
            The GEOID.
        row (_BudgetRow | None):
            Its counts, ``None`` without a budget.
        target_ids (list):
            Its targets, in ascending order of the id as a string.
        reached_ids (list):
            Its reached footprints, in the same order.
    """

    geoid: str | None
    row: _BudgetRow | None
    target_ids: list
    reached_ids: list


class _BudgetCall(NamedTuple):
    """
    The inputs of one :meth:`BlockGroupUnitBudget.allocate` call.

    Attributes:
        reached (AssetInventory):
            The reached footprints.
        target_sites (dict):
            The targets' sites.
        reached_sites (dict):
            The reached footprints' sites.
        chances (dict):
            The targets' residential probabilities.
        relabel_sources (Mapping | None):
            As :meth:`BlockGroupUnitBudget.allocate` takes it.
        state (_BudgetAllocation):
            The allocation being built.
    """

    reached: AssetInventory
    target_sites: dict
    reached_sites: dict
    chances: dict
    relabel_sources: Mapping | None
    state: _BudgetAllocation


class BlockGroupUnitBudget:
    """
    Allocate each block group's B25024 residual units to unreached footprints.

    The census table B25024 counts a block group's housing units by the
    number of units in their structure, in the structure-size bins of
    :data:`B25024_BINS` and a mobile-home line. A footprint is reached when
    an observed record gives it a class. The others, the targets, receive
    what the counts leave once the reached footprints' units are
    subtracted.

    Each target has a capacity, the whole number of units its gross floor
    area holds at :attr:`min_unit_area_sqft` square feet per unit. The gross
    floor area is the target's
    :data:`~brails.aggregators.points_to_polygons.nsi_footprint_join.GROSS_AREA`
    in square feet where it holds a positive number, else its plan area
    times its :data:`~brails.utils.world_values.STORIES_KEY` where that is
    a positive number, else its plan area alone. The candidates are the
    targets with plan area and a capacity of at least one unit.

    Each block group with a B25024 row goes through seven steps.

    1. Mobile homes. The table's mobile-home count less the units of the
       reached :data:`MOBILE_HOME_CLASS` footprints, rounded down and
       floored at zero, is the number of footprints made
       :data:`MOBILE_HOME_CLASS` with :data:`MOBILE_HOME_UNITS` unit. They
       are taken nearest first from the footprints within
       :attr:`mh_park_radius_m` of a mobile-home park point: the targets
       with a capacity of at least one unit, and the reached
       :data:`RELABEL_CLASS` footprints that ``allocate`` names in
       ``relabel_sources``, whose class is then withheld from its source.
       A reached footprint keeps its source's unit count, so its capacity
       is not checked. Mobile homes are not candidates.
    2. Residuals. Each structure-size bin's residual ``R_b`` is its
       estimate less the units of the reached footprints of its class in
       :data:`BIN_CLASSES`, other than those made mobile homes, floored at
       zero.
    3. The total deficit. Flooring at zero hides a bin whose reached units
       exceed its estimate, so the residuals can sum to more than the block
       group lacks as a whole, the estimates' sum less the reached units'
       sum. Unless :attr:`per_bin` is set, the residuals are then replaced
       by the deficit's whole units, shared among the bins in proportion to
       their whole residuals by largest remainders. No bin gains.
    4. Structures per bin. A bin needs as many structures as its residual
       fills at the middle of its unit range, ``R_b / midpoint_b``, a
       fraction, with the midpoints of
       :data:`~brails.aggregators.points_to_polygons.nsi_footprint_join.UNIT_MIDPOINTS`.
       The rounded sum of these needs, capped at the number of candidates,
       is shared among the bins in proportion to their needs by largest
       remainders. A bin then keeps no more structures than its whole
       residual fills at the bin's least unit count ``low_b``, that is
       ``floor(R_b) // low_b``, and, counting from the largest bin down, no
       more than the candidates left whose capacity holds ``low_b``.
    5. Dwellings. The candidates are ranked by the imputer's residential
       probability (highest first, unknown last), then plan area (largest
       first), then id. From the largest bin down, each bin reserves one
       candidate per structure: the highest-ranked of those left whose
       capacity holds the bin's even share (its whole residual over its
       structures, within the bin's range), then, where too few do, the
       largest that hold its least unit count, ties by rank. The reserved
       candidates are the dwellings, the same in every world.
    6. Bins and units per world. In each world the dwellings are put in a
       random order that tends to put larger floor areas first: descending
       ``ln(u) / A``, with ``u`` the dwelling's keyed uniform under
       :data:`PURPOSE_UNIT_BUDGET` and ``A`` its gross floor area in square
       meters.
       From the largest bin down, each bin's structures go to the first
       dwellings in that order whose capacity holds the bin's even share,
       then, where too few do, to the largest that hold its least unit
       count, ties in that order, which the reservation makes always
       possible. The bin's whole residual is split evenly over its
       structures, the extra units to the first, and clipped to the bin's
       range. A dwelling then holds no more than its capacity, and the
       units above it move, in order, to the bin's dwellings with capacity
       left. Which dwelling holds a bin varies by world, the units each bin
       places do not, and no bin receives more than its residual.
    7. The rest. A target that became neither a mobile home nor a dwelling
       is dropped when its plan area is under :attr:`threshold_sqft` and it
       lies beyond :attr:`mh_park_radius_m` of every mobile-home park
       point, and is otherwise passed to the non-residential stage.

    What a step cannot place is recorded in the result's ``gaps``, and
    ``allocate`` warns where structures found no candidate able to hold
    them or units exceeded their dwellings' capacity. A block group without
    a B25024 row has no budget: its targets are left to the imputer's
    residential round, and with :attr:`drop_without_acs` the small ones
    whose residential probability is below :data:`DROP_PROBABILITY` are
    dropped.

    The counts are those of the whole block group, while the budget sees
    only the footprints passed to it, so the footprints of a block group
    must be passed whole. A study area that cuts a block group receives
    that block group's whole residual in the part inside.

    Every constructor argument is stored as an attribute of the same name,
    the table and the mobile-home park points as given. ``seed`` holds the
    seed used, never ``None``.
    """

    def __init__(
        self,
        b25024: pd.DataFrame,
        mh_park_points: Any,  # noqa: ANN401
        *,
        imputer: ObservedDonorImputer,
        seed: int | None = None,
        threshold_sqft: float = 1000.0,
        mh_park_radius_m: float = 300.0,
        moe_factor: float = 2.0,
        min_unit_area_sqft: float = 300.0,
        drop_without_acs: bool = False,
        per_bin: bool = False,
    ) -> None:
        """
        Configure the budget.

        Args:
            b25024 (pandas.DataFrame):
                B25024 at block group, with a :data:`GEOID_COLUMN` column of
                GEOIDs as strings of digits and the estimate and margin
                columns of lines 2 to 10 named as the Census API names
                them, such as ``B25024_002E`` and ``B25024_002M``. A block
                group with a negative or missing estimate on any of these
                lines has no budget.
            mh_park_points (Any):
                The mobile-home park points: a GeoDataFrame or GeoSeries of
                points, whose missing CRS is read as longitude and
                latitude, or an array of longitude and latitude pairs.
            imputer (ObservedDonorImputer):
                The imputer whose residential probability ranks the
                candidates, with the default status.
            seed (int, optional):
                The seed. ``None`` draws one from system entropy.
            threshold_sqft (float, optional):
                The plan area in square feet under which a target neither
                a mobile home nor a dwelling is dropped.
            mh_park_radius_m (float, optional):
                The distance to a mobile-home park point in meters within
                which a footprint may become a mobile home and is never
                dropped.
            moe_factor (float, optional):
                The multiple of the margin of error that bounds the
                diagnostic.
            min_unit_area_sqft (float, optional):
                The gross floor area in square feet a housing unit needs,
                which sets each target's capacity.
            drop_without_acs (bool, optional):
                Whether, in a block group without a budget, a target under
                the threshold, beyond the mobile-home park radius and with a
                residential probability below :data:`DROP_PROBABILITY` is
                dropped.
            per_bin (bool, optional):
                Whether each bin's residual stands on its own instead of
                sharing the block group's total deficit.

        Raises:
            ValueError:
                If a threshold, radius or factor is negative, the gross
                area per unit is not positive, the table lacks a column,
                holds a GEOID that is not a string of
                :data:`~brails.constants.GEOID_LENGTH` digits or repeats
                one, a mobile-home park geometry is not a point, or a
                mobile-home park array is not of longitude and latitude
                pairs, or ``imputer`` reads a status other than the
                residential occupancy classes, its default.
        """
        status = (imputer.status_key, set(imputer.status_values))
        if status != (OCCUPANCY_KEY, set(UNIT_CLASSES)):
            msg = (
                'The budget ranks dwellings by the residential probability and '
                f'needs an imputer whose status is {OCCUPANCY_KEY} among '
                f'{UNIT_CLASSES}.'
            )
            raise ValueError(msg)
        if min(threshold_sqft, mh_park_radius_m, moe_factor) < 0:
            msg = (
                'threshold_sqft, mh_park_radius_m and moe_factor must not be '
                'negative.'
            )
            raise ValueError(msg)
        if not min_unit_area_sqft > 0:
            msg = 'min_unit_area_sqft must be positive.'
            raise ValueError(msg)
        self.b25024 = b25024
        self.mh_park_points = mh_park_points
        self.imputer = imputer
        self.seed = secrets.randbits(64) if seed is None else int(seed)
        self.threshold_sqft = float(threshold_sqft)
        self.mh_park_radius_m = float(mh_park_radius_m)
        self.moe_factor = float(moe_factor)
        self.min_unit_area_sqft = float(min_unit_area_sqft)
        self.drop_without_acs = bool(drop_without_acs)
        self.per_bin = bool(per_bin)
        self._rows = _read_b25024(b25024)
        mh_parks = _mh_park_lonlat(mh_park_points)
        self._mh_park_tree = (
            BallTree(np.radians(mh_parks[:, ::-1]), metric='haversine')
            if len(mh_parks)
            else None
        )

    def allocate(
        self,
        targets: AssetInventory,
        reached: AssetInventory,
        *,
        n_worlds: int = 1,
        relabel_sources: Mapping[Hashable, str] | None = None,
    ) -> UnitBudgetResult:
        """
        Allocate the budgets of the block groups the footprints lie in.

        Args:
            targets (AssetInventory):
                The unreached footprints, with one possible world, each
                holding :data:`~brails.constants.BLOCK_GROUP_KEY`.
                Not modified.
            reached (AssetInventory):
                The reached footprints with their class and unit count,
                with one possible world, each holding
                :data:`~brails.constants.BLOCK_GROUP_KEY`.
                They are also the imputer's donors. A footprint's units are
                counted as
                :func:`~brails.aggregators.points_to_polygons.nsi_footprint_join.unit_contribution`
                counts a record's. Not modified.
            n_worlds (int, optional):
                The number of possible worlds. Defaults to 1.
            relabel_sources (Mapping[Hashable, str], optional):
                The reached footprints the mobile-home rule may reclassify,
                each with the label of the source holding its class, which
                the withheld row names. Only those of class
                :data:`RELABEL_CLASS` are eligible. ``None``, the default,
                admits none.

        Returns:
            UnitBudgetResult:
                The allocation.

        Raises:
            ValueError:
                If ``n_worlds`` is below 1, an inventory holds more than
                one possible world, the two share an asset, or a
                footprint's coordinates are not a footprint
                :meth:`AssetInventory.has_area` reads.

        Warns:
            UserWarning:
                Once, if a bin's structures found no candidate able to hold
                them or its units exceeded its dwellings' capacity, naming
                the block groups and units concerned.
        """
        if n_worlds < 1:
            msg = 'n_worlds must be at least 1.'
            raise ValueError(msg)
        for name, inventory in (('targets', targets), ('reached', reached)):
            if inventory.n_pw != 1:
                msg = f'The {name} inventory must hold one possible world.'
                raise ValueError(msg)
        shared = sorted(set(targets.inventory) & set(reached.inventory), key=str)
        if shared:
            msg = f'Targets and reached footprints share assets: {shared[:10]}.'
            raise ValueError(msg)

        target_sites = self._sites(targets)
        call = _BudgetCall(
            reached,
            target_sites,
            self._sites(reached),
            self._chances(targets, reached, target_sites),
            relabel_sources,
            _BudgetAllocation(n_worlds),
        )
        for group in self._groups(call):
            if group.row is None:
                self._no_budget(call, group)
                continue
            mobile = self._mobile_homes(call, group)
            reached_units, residual = self._bin_residuals(call, group, mobile)
            candidates = self._candidates(call, group, mobile)
            structures, dwellings = self._structures(
                call, group, residual, candidates
            )
            generated = self._assign(call, group, dwellings, structures, residual)
            self._settle(call, group, mobile, candidates)
            self._diagnose_bins(call, group, reached_units, generated)

        over = [g for g in call.state.gaps if g['kind'] == GAP_UNITS_OVER_CAPACITY]
        short = [g for g in call.state.gaps if g['kind'] == GAP_CANDIDATE_SHORTFALL]
        if over or short:
            structures = sum(g['count'] for g in short)
            units = sum(g['units'] for g in over)
            msg = (
                f'In {len({g["block_group"] for g in over + short})} of the block '
                f'groups, {structures} structure{"s" * (structures != 1)} found no '
                f'footprint able to hold {"them" if structures != 1 else "it"} '
                f'and up to {units} unit{"s" * (units != 1)} found no dwelling of '
                "their bin with room, as the result's gaps record under "
                f'{GAP_CANDIDATE_SHORTFALL} and {GAP_UNITS_OVER_CAPACITY}.'
            )
            warnings.warn(msg, stacklevel=2)
        return call.state.result(targets, reached, self.seed)

    # ------------------------------------------------------------------
    # Inputs

    def _sites(self, inventory: AssetInventory) -> dict[Hashable, _BudgetSite]:
        """
        Locate and measure every asset.

        The measures are the plan area, the capacity at
        :attr:`min_unit_area_sqft` per unit and the distance to the nearest
        mobile-home park point.

        Args:
            inventory (AssetInventory):
                The assets.

        Returns:
            dict[Hashable, _BudgetSite]:
                The site per asset.

        Raises:
            ValueError:
                If an asset's coordinates are not a footprint
                :meth:`AssetInventory.has_area` reads.
        """
        ids, lonlat, sites = [], [], {}
        for asset_id, asset in inventory.inventory.items():
            block_group = asset.features.get(BLOCK_GROUP_KEY)
            if not isinstance(block_group, str) or is_missing_value(block_group):
                block_group = None
            try:
                with_area = AssetInventory.has_area(asset.coordinates)
            except ValueError as error:
                msg = f'Asset {asset_id!r} has no readable footprint: {error}'
                raise ValueError(msg) from error
            area = 0.0
            if with_area:
                lon, lat, area = AssetInventory.centroid_and_plan_area(
                    asset.coordinates
                )
                ids.append(asset_id)
                lonlat.append((lat, lon))
            # The gross floor area: GrossArea, else plan area times stories,
            # else plan area, each where a positive finite number is held.
            held = {
                key: float(value)
                for key in (GROSS_AREA, STORIES_KEY)
                if isinstance(
                    value := asset.features.get(key),
                    (int, float, np.integer, np.floating),
                )
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
            }
            stories = None if GROSS_AREA in held else held.get(STORIES_KEY)
            gross_sqft = held.get(
                GROSS_AREA, area / UNIT_TO_METER['ft'] ** 2 * (stories or 1.0)
            )
            source = (
                AREA_FROM_GROSS_AREA
                if GROSS_AREA in held
                else AREA_FROM_PLAN_AREA
                if stories is None
                else AREA_FROM_STORIES
            )
            sites[asset_id] = _BudgetSite(
                area,
                block_group,
                math.inf,
                gross_sqft * UNIT_TO_METER['ft'] ** 2,
                math.floor(gross_sqft / self.min_unit_area_sqft),
                source,
                math.nan if stories is None else stories,
            )
        if ids and self._mh_park_tree is not None:
            radians, _ = self._mh_park_tree.query(np.radians(lonlat), k=1)
            for asset_id, distance in zip(ids, radians[:, 0] * EARTH_RADIUS_M):
                sites[asset_id] = sites[asset_id]._replace(mh_park_m=float(distance))
        return sites

    def _chances(
        self,
        targets: AssetInventory,
        reached: AssetInventory,
        sites: Mapping[Hashable, _BudgetSite],
    ) -> dict[Hashable, float]:
        """
        Compute the residential probability of every target that may need it.

        Args:
            targets (AssetInventory):
                The targets.
            reached (AssetInventory):
                The donors.
            sites (Mapping[Hashable, _BudgetSite]):
                The targets' sites.

        Returns:
            dict[Hashable, float]:
                The probability per target with area in a block group with
                a budget, and, with :attr:`drop_without_acs`, per small
                target without one.
        """
        wanted = [
            asset_id
            for asset_id, site in sites.items()
            if site.area_m2 > 0
            and (
                site.block_group in self._rows
                or (self.drop_without_acs and self._is_small(site))
            )
        ]
        if not wanted:
            return {}
        pool = self.imputer.pool(reached, targets)
        return self.imputer.status_probability(pool, sorted(wanted, key=str))

    def _groups(self, call: _BudgetCall) -> list[_BudgetGroup]:
        """
        Group the footprints by block group.

        Args:
            call (_BudgetCall):
                The call.

        Returns:
            list[_BudgetGroup]:
                Every block group holding a target, and every block group
                with a budget holding a reached footprint, in ascending
                order of the GEOID as a string.
        """
        members: dict[str | None, tuple[list, list]] = {}
        for asset_id in sorted(call.target_sites, key=str):
            geoid = call.target_sites[asset_id].block_group
            members.setdefault(geoid, ([], []))[0].append(asset_id)
        for asset_id in sorted(call.reached_sites, key=str):
            geoid = call.reached_sites[asset_id].block_group
            if geoid in members or geoid in self._rows:
                members.setdefault(geoid, ([], []))[1].append(asset_id)
        return [
            _BudgetGroup(
                geoid,
                self._rows.get(geoid) if geoid is not None else None,
                *members[geoid],
            )
            for geoid in sorted(members, key=str)
        ]

    # ------------------------------------------------------------------
    # Steps per block group

    def _no_budget(self, call: _BudgetCall, group: _BudgetGroup) -> None:
        """
        Log the targets of a block group without a budget.

        The small ones are dropped under :attr:`drop_without_acs`, and the
        rest are left to the imputer.

        Args:
            call (_BudgetCall):
                The call.
            group (_BudgetGroup):
                The block group.
        """
        state = call.state
        left = 0
        for asset_id in group.target_ids:
            site = call.target_sites[asset_id]
            chance = call.chances.get(asset_id, math.nan)
            if (
                self.drop_without_acs
                and self._is_small(site)
                and chance < DROP_PROBABILITY
            ):
                state.drop(asset_id, site, chance)
            else:
                state.log_target(asset_id, site, OUTCOME_NO_ACS, chance)
                left += 1
        if group.target_ids:
            state.record_gap(group.geoid, None, GAP_NO_ACS, left, math.nan)

    def _mobile_homes(self, call: _BudgetCall, group: _BudgetGroup) -> set:
        """
        Place a block group's mobile homes and diagnose their bin.

        Args:
            call (_BudgetCall):
                The call.
            group (_BudgetGroup):
                The block group.

        Returns:
            set:
                The targets and reached footprints made mobile homes.
        """
        reached_units = math.fsum(
            units
            for occupancy, units in map(
                self._class_and_units(call), group.reached_ids
            )
            if occupancy == MOBILE_HOME_CLASS
        )
        residual = math.floor(
            max(0.0, group.row.estimate[MOBILE_HOME_LABEL] - reached_units)
        )
        eligible = [
            (call.target_sites[a].mh_park_m, str(a), a, False)
            for a in group.target_ids
            if call.target_sites[a].mh_park_m <= self.mh_park_radius_m
            and call.target_sites[a].capacity >= 1
        ]
        relabel_sources = call.relabel_sources or {}
        for asset_id in group.reached_ids:
            if asset_id not in relabel_sources:
                continue
            occupancy, _ = self._class_and_units(call)(asset_id)
            site = call.reached_sites[asset_id]
            if (
                occupancy == RELABEL_CLASS
                and site.mh_park_m <= self.mh_park_radius_m
            ):
                eligible.append((site.mh_park_m, str(asset_id), asset_id, True))
        eligible.sort(key=lambda item: (item[0], item[1]))
        chosen = eligible[:residual]
        for _, _, asset_id, is_reached in chosen:
            if is_reached:
                call.state.add_relabeled(
                    asset_id,
                    call.reached_sites[asset_id],
                    relabel_sources[asset_id],
                    RELABEL_CLASS,
                )
            else:
                call.state.add_mobile_home(asset_id, call.target_sites[asset_id])
        if residual > len(chosen):
            short = residual - len(chosen)
            call.state.record_gap(
                group.geoid,
                MOBILE_HOME_LABEL,
                GAP_MOBILE_HOME_SHORTFALL,
                short,
                short,
            )
        row = group.row
        call.state.diagnose(
            group.geoid,
            MOBILE_HOME_LABEL,
            row.estimate[MOBILE_HOME_LABEL],
            row.margin[MOBILE_HOME_LABEL],
            reached_units,
            [float(len(chosen) * MOBILE_HOME_UNITS)] * call.state.n_worlds,
            self.moe_factor,
        )
        return {asset_id for _, _, asset_id, _ in chosen}

    def _bin_residuals(
        self, call: _BudgetCall, group: _BudgetGroup, mobile: set
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute each structure-size bin's reached units and residual.

        Args:
            call (_BudgetCall):
                The call.
            group (_BudgetGroup):
                The block group.
            mobile (set):
                The footprints made mobile homes, whose units do not count.

        Returns:
            tuple[dict[str, float], dict[str, float]]:
                The reached units and the residual per bin label, the
                residuals shared out of the total deficit unless
                :attr:`per_bin` is set.
        """
        bin_of_class = {occupancy: b for b, occupancy in BIN_CLASSES.items()}
        reached_units = dict.fromkeys(B25024_BINS, 0.0)
        for asset_id in group.reached_ids:
            if asset_id in mobile:
                continue
            occupancy, units = self._class_and_units(call)(asset_id)
            if occupancy in bin_of_class:
                reached_units[bin_of_class[occupancy]] += units
        estimate = group.row.estimate
        residual = {b: max(0.0, estimate[b] - reached_units[b]) for b in B25024_BINS}
        if not self.per_bin:
            deficit = max(
                0.0,
                math.fsum(estimate[b] for b in B25024_BINS)
                - math.fsum(reached_units.values()),
            )
            if math.fsum(residual.values()) > deficit:
                whole = [math.floor(residual[b]) for b in B25024_BINS]
                shared = _apportion(whole, min(math.floor(deficit), sum(whole)))
                residual = dict(zip(B25024_BINS, map(float, shared)))
        return reached_units, residual

    def _candidates(
        self, call: _BudgetCall, group: _BudgetGroup, mobile: set
    ) -> list:
        """
        Rank the targets that may become dwellings.

        Args:
            call (_BudgetCall):
                The call.
            group (_BudgetGroup):
                The block group.
            mobile (set):
                The footprints made mobile homes.

        Returns:
            list:
                The targets with area and a capacity of at least one unit
                that are not mobile homes, by residential probability
                descending with NaN last, then plan area descending, then
                the id as a string.
        """
        candidates = [
            a
            for a in group.target_ids
            if a not in mobile
            and call.target_sites[a].area_m2 > 0
            and call.target_sites[a].capacity >= 1
        ]

        def rank_key(asset_id: Hashable) -> tuple[bool, float, float, str]:
            chance = call.chances.get(asset_id, math.nan)
            unknown = math.isnan(chance)
            area = call.target_sites[asset_id].area_m2
            return (unknown, 0.0 if unknown else -chance, -area, str(asset_id))

        return sorted(candidates, key=rank_key)

    def _structures(
        self,
        call: _BudgetCall,
        group: _BudgetGroup,
        residual: Mapping[str, float],
        candidates: list,
    ) -> tuple[dict[str, int], list]:
        """
        Decide each bin's structures and reserve the dwellings that hold them.

        Args:
            call (_BudgetCall):
                The call.
            group (_BudgetGroup):
                The block group.
            residual (Mapping[str, float]):
                The residual per bin.
            candidates (list):
                The ranked candidates.

        Returns:
            tuple[dict[str, int], list]:
                The structures per bin label and the dwellings in the
                candidates' order, both the same in every world.
        """
        labels = list(B25024_BINS)
        # The midpoints of the Hazus classes: README.rst beside this module,
        # "The census table and the classes".
        needed = [residual[b] / UNIT_MIDPOINTS[BIN_CLASSES[b]] for b in labels]
        # half up, not round()'s half to even
        wanted = math.floor(math.fsum(needed) + 0.5)
        total = min(wanted, len(candidates))
        state = call.state
        if wanted > total:
            state.record_gap(
                group.geoid, None, GAP_CANDIDATE_SHORTFALL, wanted - total, math.nan
            )
        structures = dict(zip(labels, _apportion(needed, total)))
        left, reserved = list(candidates), set()
        for b in reversed(labels):
            whole = math.floor(residual[b])
            low, high = _class_range(BIN_CLASSES[b])
            kept = min(structures[b], whole // low)
            if kept < structures[b]:
                state.record_gap(
                    group.geoid,
                    b,
                    GAP_STRUCTURES_WITHOUT_UNITS,
                    structures[b] - kept,
                    math.nan,
                )
            # The highest-ranked candidates left whose capacity holds the
            # bin's even share over the structures it can place, then the
            # largest of those that hold its least unit count.
            capacity = {a: call.target_sites[a].capacity for a in left}
            count = min(kept, sum(capacity[a] >= low for a in left))
            top = whole if high is None else high
            share = min(math.ceil(whole / count), top) if count else low
            fit = [a for a in left if capacity[a] >= share]
            rest = [a for a in left if low <= capacity[a] < share]
            held = (fit + sorted(rest, key=lambda a: -capacity[a]))[:count]
            if len(held) < kept:
                state.record_gap(
                    group.geoid,
                    b,
                    GAP_CANDIDATE_SHORTFALL,
                    kept - len(held),
                    math.nan,
                )
            structures[b] = len(held)
            reserved.update(held)
            left = [a for a in left if a not in reserved]
            if not held and whole > 0:
                state.record_gap(group.geoid, b, GAP_BIN_WITHOUT_STRUCTURE, 0, whole)
        return structures, [a for a in candidates if a in reserved]

    def _assign(
        self,
        call: _BudgetCall,
        group: _BudgetGroup,
        dwellings: list,
        structures: Mapping[str, int],
        residual: Mapping[str, float],
    ) -> list[dict[str, int]]:
        """
        Give the dwellings their bins and units in every world.

        Args:
            call (_BudgetCall):
                The call.
            group (_BudgetGroup):
                The block group.
            dwellings (list):
                The dwellings.
            structures (Mapping[str, int]):
                The structures per bin.
            residual (Mapping[str, float]):
                The residual per bin.

        Returns:
            list[dict[str, int]]:
                Per world, the units placed per bin.
        """
        shares, tops = {}, {}
        for b, count in structures.items():
            whole = math.floor(residual[b])
            _, high = _class_range(BIN_CLASSES[b])
            tops[b] = whole if high is None else high
            base, extra = divmod(whole, count) if count else (0, 0)
            even = [base + (1 if i < extra else 0) for i in range(count)]
            shares[b] = np.minimum(np.array(even, dtype=int), tops[b])
            if count and whole > shares[b].sum():
                call.state.record_gap(
                    group.geoid,
                    b,
                    GAP_UNITS_CLIPPED,
                    0,
                    whole - int(shares[b].sum()),
                )
        placed = [dict.fromkeys(B25024_BINS, 0) for _ in range(call.state.n_worlds)]
        if not dwellings:
            return placed
        capacity = np.array([call.target_sites[a].capacity for a in dwellings])
        areas = np.array([call.target_sites[a].gross_m2 for a in dwellings])
        keys = asset_key_digest(dwellings)
        over = dict.fromkeys(B25024_BINS, 0)
        for world, in_world in enumerate(placed):
            u = keyed_uniforms(
                keys, seed=self.seed, purpose=PURPOSE_UNIT_BUDGET, world=world
            )
            with np.errstate(divide='ignore'):
                score = np.log(u) / areas
            free = sorted(
                range(len(dwellings)), key=lambda i: (-score[i], str(dwellings[i]))
            )
            for b in reversed(B25024_BINS):
                # The first dwellings whose capacity holds the bin's even
                # share, then the largest of those that hold its least unit
                # count.
                low, _ = _class_range(BIN_CLASSES[b])
                share = int(shares[b].max(initial=low))
                fit = [i for i in free if capacity[i] >= share]
                rest = [i for i in free if low <= capacity[i] < share]
                rest.sort(key=lambda i: -capacity[i])
                chosen = (fit + rest)[: structures[b]]
                # The reservation leaves every bin enough dwellings to hold it.
                assert len(chosen) == structures[b], f'bin {b} lacks dwellings'
                free = [i for i in free if i not in set(chosen)]
                limit = np.minimum(capacity[chosen], tops[b])
                units = np.minimum(shares[b], limit)
                # The units above a dwelling's capacity move, in order, to the
                # dwellings with capacity left.
                room = limit - units
                excess = int(shares[b].sum() - units.sum())
                units += np.clip(excess - (np.cumsum(room) - room), 0, room)
                over[b] = max(over[b], excess - int(room.sum()))
                for i, count in zip(chosen, units):
                    call.state.assign(
                        dwellings[i], world, BIN_CLASSES[b], int(count)
                    )
                in_world[b] = int(units.sum())
        for b in B25024_BINS:
            if over[b] > 0:
                call.state.record_gap(
                    group.geoid, b, GAP_UNITS_OVER_CAPACITY, 0, over[b]
                )
        return placed

    def _settle(
        self, call: _BudgetCall, group: _BudgetGroup, mobile: set, candidates: list
    ) -> None:
        """
        Log every target of a block group with a budget, dropping or passing on.

        Args:
            call (_BudgetCall):
                The call.
            group (_BudgetGroup):
                The block group.
            mobile (set):
                The footprints made mobile homes.
            candidates (list):
                The ranked candidates.
        """
        state = call.state
        ranks = {a: i + 1 for i, a in enumerate(candidates)}
        for asset_id in group.target_ids:
            if asset_id in mobile:
                continue
            site = call.target_sites[asset_id]
            chance = call.chances.get(asset_id, math.nan)
            rank = ranks.get(asset_id)
            if asset_id in state.classes:
                state.log_target(asset_id, site, OUTCOME_DWELLING, chance, rank)
                state.residential[asset_id] = True
            elif self._is_small(site):
                state.drop(asset_id, site, chance, rank)
            else:
                state.pass_on(asset_id, site, chance, rank)

    def _diagnose_bins(
        self,
        call: _BudgetCall,
        group: _BudgetGroup,
        reached_units: Mapping[str, float],
        generated: Sequence[Mapping[str, int]],
    ) -> None:
        """
        Compare each structure-size bin and their total with the estimate.

        Args:
            call (_BudgetCall):
                The call.
            group (_BudgetGroup):
                The block group.
            reached_units (Mapping[str, float]):
                The reached units per bin.
            generated (Sequence[Mapping[str, int]]):
                Per world, the units placed per bin.
        """
        row = group.row
        for b in B25024_BINS:
            call.state.diagnose(
                group.geoid,
                b,
                row.estimate[b],
                row.margin[b],
                reached_units[b],
                [float(in_world[b]) for in_world in generated],
                self.moe_factor,
            )
        call.state.diagnose(
            group.geoid,
            TOTAL_LABEL,
            math.fsum(row.estimate[b] for b in B25024_BINS),
            _sum_margin([row.margin[b] for b in B25024_BINS]),
            math.fsum(reached_units.values()),
            [float(sum(in_world.values())) for in_world in generated],
            self.moe_factor,
        )

    # ------------------------------------------------------------------
    # Helpers

    @staticmethod
    def _class_and_units(call: _BudgetCall) -> Any:  # noqa: ANN401
        """
        Return a reader of a reached footprint's class and counted units.

        Args:
            call (_BudgetCall):
                The call.

        Returns:
            Any:
                A function of an asset id returning its class, ``None``
                where missing, and its units as
                :func:`~brails.aggregators.points_to_polygons.nsi_footprint_join.unit_contribution`
                counts them, zero for a class outside the housing classes.
        """

        def read(asset_id: Hashable) -> tuple[str | None, float]:
            features = call.reached.inventory[asset_id].features
            occupancy = features.get(OCCUPANCY_KEY)
            if not isinstance(occupancy, str) or is_missing_value(occupancy):
                return None, 0.0
            units, _ = unit_contribution(occupancy, features.get(UNITS_KEY))
            return occupancy, 0.0 if units is None else float(units)

        return read

    def _is_small(self, site: _BudgetSite) -> bool:
        """
        Return whether a site may be dropped: small, beyond every mobile-home park.

        Args:
            site (_BudgetSite):
                The site.

        Returns:
            bool:
                Whether its plan area is under :attr:`threshold_sqft` and it
                lies beyond :attr:`mh_park_radius_m` of every mobile-home
                park point.
        """
        return (
            site.area_m2 / UNIT_TO_METER['ft'] ** 2 < self.threshold_sqft
            and not site.mh_park_m <= self.mh_park_radius_m
        )


class _BudgetAllocation:
    """
    The allocation under construction, shared by the steps of one call.

    Attributes:
        n_worlds (int):
            The number of worlds.
        classes (dict):
            Per asset given a class, the class per world.
        units (dict):
            Per asset given a unit count, the count per world.
        residential (dict):
            The residential status per target.

    The other attributes collect the rows and ids of the result's fields of
    the same names.
    """

    def __init__(self, n_worlds: int) -> None:
        """
        Start an empty allocation.

        Args:
            n_worlds (int):
                The number of worlds.
        """
        self.n_worlds = n_worlds
        self.classes: dict[Hashable, list] = {}
        self.units: dict[Hashable, list] = {}
        self.residential: dict[Hashable, bool] = {}
        self.withheld: list[dict[str, Any]] = []
        self.dropped: list = []
        self.nonresidential: list = []
        self.log: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []
        self.diagnostic: list[dict[str, Any]] = []

    def assign(
        self, asset_id: Hashable, world: int, occupancy: str, units: int
    ) -> None:
        """
        Give a dwelling its class and units in one world.

        Args:
            asset_id (Hashable):
                The dwelling.
            world (int):
                The world.
            occupancy (str):
                The class.
            units (int):
                The unit count.
        """
        self.classes.setdefault(asset_id, [None] * self.n_worlds)[world] = occupancy
        self.units.setdefault(asset_id, [None] * self.n_worlds)[world] = units

    def add_mobile_home(self, asset_id: Hashable, site: _BudgetSite) -> None:
        """
        Make a target a mobile home in every world.

        Args:
            asset_id (Hashable):
                The target.
            site (_BudgetSite):
                Its site.
        """
        self.classes[asset_id] = [MOBILE_HOME_CLASS] * self.n_worlds
        self.units[asset_id] = [MOBILE_HOME_UNITS] * self.n_worlds
        self.residential[asset_id] = True
        self.log_target(asset_id, site, OUTCOME_MOBILE_HOME, math.nan)

    def add_relabeled(
        self, asset_id: Hashable, site: _BudgetSite, source: str, previous: str
    ) -> None:
        """
        Make a reached footprint a mobile home and withhold its class.

        Its unit count is left to its source.

        Args:
            asset_id (Hashable):
                The reached footprint.
            site (_BudgetSite):
                Its site.
            source (str):
                The label of the source holding its class.
            previous (str):
                The class withheld.
        """
        self.classes[asset_id] = [MOBILE_HOME_CLASS] * self.n_worlds
        self.withheld.append(
            dict(zip(WITHHELD_COLUMNS, (asset_id, OCCUPANCY_KEY, source, previous)))
        )
        self.log_target(asset_id, site, OUTCOME_RELABELED, math.nan, reached=True)

    def drop(
        self,
        asset_id: Hashable,
        site: _BudgetSite,
        chance: float,
        rank: int | None = None,
    ) -> None:
        """
        Drop a target.

        Args:
            asset_id (Hashable):
                The target.
            site (_BudgetSite):
                Its site.
            chance (float):
                Its residential probability.
            rank (int | None, optional):
                Its candidacy rank.
        """
        self.dropped.append(asset_id)
        self.log_target(asset_id, site, OUTCOME_DROPPED, chance, rank)

    def pass_on(
        self,
        asset_id: Hashable,
        site: _BudgetSite,
        chance: float,
        rank: int | None = None,
    ) -> None:
        """
        Pass a target to the non-residential stage.

        Args:
            asset_id (Hashable):
                The target.
            site (_BudgetSite):
                Its site.
            chance (float):
                Its residential probability.
            rank (int | None, optional):
                Its candidacy rank.
        """
        self.nonresidential.append(asset_id)
        self.residential[asset_id] = False
        self.log_target(asset_id, site, OUTCOME_NONRESIDENTIAL, chance, rank)

    def log_target(
        self,
        asset_id: Hashable,
        site: _BudgetSite,
        outcome: str,
        chance: float,
        rank: int | None = None,
        *,
        reached: bool = False,
    ) -> None:
        """
        Record the outcome of one footprint.

        Args:
            asset_id (Hashable):
                The footprint.
            site (_BudgetSite):
                Its site.
            outcome (str):
                The outcome.
            chance (float):
                Its residential probability, NaN where not computed.
            rank (int | None, optional):
                Its candidacy rank.
            reached (bool, optional):
                Whether it is a reached footprint.
        """
        values = (
            asset_id,
            site.block_group,
            reached,
            outcome,
            chance,
            math.nan if rank is None else rank,
            site.area_m2 / UNIT_TO_METER['ft'] ** 2,
            site.gross_source,
            site.stories,
            site.capacity,
            site.mh_park_m,
        )
        self.log.append(dict(zip(LOG_COLUMNS, values)))

    def record_gap(
        self,
        block_group: str | None,
        label: str | None,
        kind: str,
        count: int,
        units: float,
    ) -> None:
        """
        Record a part of a budget left unplaced.

        Args:
            block_group (str | None):
                The block group.
            label (str | None):
                The bin, ``None`` for the whole block group.
            kind (str):
                The kind of gap.
            count (int):
                The structures concerned, or the targets for
                :data:`GAP_NO_ACS`.
            units (float):
                The units concerned, NaN where not counted.
        """
        self.gaps.append(
            dict(zip(GAP_COLUMNS, (block_group, label, kind, count, units)))
        )

    def diagnose(
        self,
        block_group: str,
        label: str,
        estimate: float,
        margin: float,
        reached_units: float,
        generated: Sequence[float],
        factor: float,
    ) -> None:
        """
        Compare one bin's units with the census estimate in every world.

        Args:
            block_group (str):
                The block group.
            label (str):
                The bin.
            estimate (float):
                The estimate.
            margin (float):
                Its margin of error.
            reached_units (float):
                The units of the reached footprints of the bin.
            generated (Sequence[float]):
                Per world, the units the budget placed in the bin.
            factor (float):
                The multiple of the margin.
        """
        lower = estimate - factor * margin
        upper = estimate + factor * margin
        for world, placed in enumerate(generated):
            total = reached_units + placed
            values = (
                block_group,
                world,
                label,
                estimate,
                margin,
                reached_units,
                placed,
                total,
                lower,
                upper,
                bool(total < lower or total > upper),
            )
            self.diagnostic.append(dict(zip(DIAGNOSTIC_COLUMNS, values)))

    def result(
        self, targets: AssetInventory, reached: AssetInventory, seed: int
    ) -> UnitBudgetResult:
        """
        Assemble the result.

        Args:
            targets (AssetInventory):
                The targets, for coordinates.
            reached (AssetInventory):
                The reached footprints, for coordinates.
            seed (int):
                The seed.

        Returns:
            UnitBudgetResult:
                The allocation.
        """
        imputed = AssetInventory()
        for asset_id in sorted(self.classes, key=str):
            features = {OCCUPANCY_KEY: collapse_realizations(self.classes[asset_id])}
            if asset_id in self.units:
                features[UNITS_KEY] = collapse_realizations(self.units[asset_id])
            holder = targets if asset_id in targets.inventory else reached
            coordinates = deepcopy(holder.inventory[asset_id].coordinates)
            imputed.add_asset(asset_id, Asset(asset_id, coordinates, features))
        imputed.n_pw = self.n_worlds
        gaps = pd.DataFrame(self.gaps, columns=list(GAP_COLUMNS))
        for column in ('block_group', 'bin'):
            values = gaps[column].astype(object)
            gaps[column] = values.where(values.notna(), None)
        return UnitBudgetResult(
            imputed,
            pd.DataFrame(self.withheld, columns=list(WITHHELD_COLUMNS)),
            dict(self.residential),
            tuple(sorted(self.dropped, key=str)),
            tuple(sorted(self.nonresidential, key=str)),
            pd.DataFrame(self.diagnostic, columns=list(DIAGNOSTIC_COLUMNS)),
            gaps,
            pd.DataFrame(self.log, columns=list(LOG_COLUMNS)),
            seed,
        )


def _read_b25024(table: pd.DataFrame) -> dict[str, _BudgetRow]:
    """
    Read the counts of each block group with a budget.

    A margin that is negative or missing reads as zero. The margin of a bin
    that sums two lines is :func:`_sum_margin` of theirs. A block group
    with a negative or missing estimate on any line read has no budget.

    Args:
        table (pandas.DataFrame):
            The table, as :class:`BlockGroupUnitBudget` takes it.

    Returns:
        dict[str, _BudgetRow]:
            The counts per GEOID.

    Raises:
        ValueError:
            If a column is missing, a GEOID is not a string of
            :data:`~brails.constants.GEOID_LENGTH` digits, or
            a GEOID repeats.
    """
    lines = dict(B25024_BINS)
    lines[MOBILE_HOME_LABEL] = (MOBILE_HOME_BIN,)
    columns = [GEOID_COLUMN] + [
        f'{B25024_TABLE}_{line}{kind}'
        for parts in lines.values()
        for line in parts
        for kind in 'EM'
    ]
    missing = [c for c in columns if c not in table.columns]
    if missing:
        msg = f'The B25024 table lacks the columns {missing}.'
        raise ValueError(msg)
    geoids = table[GEOID_COLUMN].tolist()
    malformed = [
        g
        for g in geoids
        if not (isinstance(g, str) and len(g) == GEOID_LENGTH and g.isdigit())
    ]
    if malformed:
        msg = (
            f'The B25024 table holds GEOIDs that are not strings of '
            f'{GEOID_LENGTH} digits, such as {malformed[:3]}. Read the table '
            f'with dtype=str for {GEOID_COLUMN!r}.'
        )
        raise ValueError(msg)
    repeated = sorted({g for g in geoids if geoids.count(g) > 1})
    if repeated:
        msg = f'The B25024 table repeats GEOIDs: {repeated[:10]}.'
        raise ValueError(msg)
    values = {
        c: pd.to_numeric(table[c], errors='coerce').to_numpy(dtype=float)
        for c in columns[1:]
    }
    rows = {}
    for i, geoid in enumerate(geoids):
        estimate, margin = {}, {}
        for label, parts in lines.items():
            estimates = [values[f'{B25024_TABLE}_{line}E'][i] for line in parts]
            if any(not e >= 0 for e in estimates):
                break
            estimate[label] = math.fsum(estimates)
            margin[label] = _sum_margin(
                [values[f'{B25024_TABLE}_{line}M'][i] for line in parts]
            )
        else:
            rows[geoid] = _BudgetRow(estimate, margin)
    return rows


def _sum_margin(margins: list[float]) -> float:
    """
    Return the margin of error of a sum of estimates.

    The margin is the root of the summed squared margins, the
    approximation of U.S. Census Bureau, Understanding and Using American
    Community Survey Data: What All Data Users Need to Know (2020),
    chapter 8, formula (1). A negative or missing margin counts as zero.

    Args:
        margins (list[float]):
            The margins of the summed estimates.

    Returns:
        float:
            The margin of the sum.
    """
    return math.sqrt(math.fsum(m * m for m in margins if m > 0))


def _mh_park_lonlat(points: Any) -> np.ndarray:  # noqa: ANN401
    """
    Read mobile-home park points as longitude and latitude.

    Args:
        points (Any):
            As :class:`BlockGroupUnitBudget` takes them.

    Returns:
        numpy.ndarray:
            An array of shape ``(n, 2)``.

    Raises:
        ValueError:
            If a geometry is not a point, or an array is not of pairs.
    """
    geometry = getattr(points, 'geometry', None)
    if geometry is not None:
        if geometry.crs is not None:
            geometry = geometry.to_crs(GEOGRAPHIC_CRS)
        geometry = geometry[~geometry.is_empty & geometry.notna()]
        if not (geometry.geom_type == 'Point').all():
            msg = 'Every mobile-home park geometry must be a point.'
            raise ValueError(msg)
        return np.column_stack([geometry.x.to_numpy(), geometry.y.to_numpy()])
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.zeros((0, 2))
    # A two-dimensional array of longitude and latitude columns.
    if array.ndim != 2 or array.shape[1] != 2:  # noqa: PLR2004
        msg = 'Mobile-home park points must be longitude and latitude pairs.'
        raise ValueError(msg)
    return array


def _apportion(shares: list[float], total: int) -> list[int]:
    """
    Apportion a whole number to shares by largest remainders.

    Args:
        shares (list[float]):
            The shares, not negative.
        total (int):
            The number apportioned.

    Returns:
        list[int]:
            The parts, summing to ``total``, or zeros when every share is
            zero. Equal remainders go to the earlier share.
    """
    weight = math.fsum(shares)
    if total <= 0 or weight <= 0:
        return [0] * len(shares)
    quotas = [s * total / weight for s in shares]
    parts = [math.floor(q) for q in quotas]
    order = sorted(range(len(shares)), key=lambda i: (-(quotas[i] - parts[i]), i))
    for i in order[: total - sum(parts)]:
        parts[i] += 1
    return parts
