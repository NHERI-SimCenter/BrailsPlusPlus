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
Sample construction years from Census tables of the block group and tract.

.. autosummary::

      ACSYearBuiltSampler
      size_class
"""

from __future__ import annotations

import math
import secrets
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from brails.constants import BLOCK_GROUP_KEY, GEOID_LENGTH, TRACT_GEOID_LENGTH
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

# First year of the open oldest B25034 bin, a placeholder; see the README
# beside this module, "Defaults and scope".
OLDEST_YEAR = 1890

# Last year of the open newest B25034 bin: the final survey year of the
# ACS 2024 five-year estimates (2020 to 2024).
NEWEST_YEAR = 2024

# ACS 2024 five-year estimates, table B25034 "Year Structure Built": its
# ten bins as (label, first year, last year, estimate variable),
# oldest first. The eight inner bins are decades and the two open ones
# are not.
B25034_BINS = (
    ('1939 or earlier', OLDEST_YEAR, 1939, 'B25034_011E'),
    ('1940 to 1949', 1940, 1949, 'B25034_010E'),
    ('1950 to 1959', 1950, 1959, 'B25034_009E'),
    ('1960 to 1969', 1960, 1969, 'B25034_008E'),
    ('1970 to 1979', 1970, 1979, 'B25034_007E'),
    ('1980 to 1989', 1980, 1989, 'B25034_006E'),
    ('1990 to 1999', 1990, 1999, 'B25034_005E'),
    ('2000 to 2009', 2000, 2009, 'B25034_004E'),
    ('2010 to 2019', 2010, 2019, 'B25034_003E'),
    ('2020 or later', 2020, NEWEST_YEAR, 'B25034_002E'),
)

# ACS 2024 five-year estimates, table B25127 "Tenure by Year Structure
# Built by Units in Structure": its six year bins and the B25034 bins
# each holds, oldest first.
COARSE_YEARS = (
    ('1939 or earlier', ('1939 or earlier',)),
    ('1940 to 1959', ('1940 to 1949', '1950 to 1959')),
    ('1960 to 1979', ('1960 to 1969', '1970 to 1979')),
    ('1980 to 1999', ('1980 to 1989', '1990 to 1999')),
    ('2000 to 2019', ('2000 to 2009', '2010 to 2019')),
    ('2020 or later', ('2020 or later',)),
)

# B25127's six units-in-structure classes, in this order. Its labels
# are "1, detached or attached", "2 to 4", "5 to 19", "20 to 49", "50 or
# more" and "Mobile home, boat, RV, van, etc.".
SIZE_CLASSES = (
    '1 unit',
    '2 to 4',
    '5 to 19',
    '20 to 49',
    '50 or more',
    'Manufactured home',
)

# Inclusive upper unit bounds of the classes of SIZE_CLASSES before
# '50 or more', from the same B25127 labels. A count above the last bound
# is in '50 or more'.
SIZE_CLASS_BOUNDS = (1, 4, 19, 49)

# The occupancy class placed in the manufactured-home class of
# SIZE_CLASSES whatever its unit count: RES2, "Mobile Home" in Hazus
# Inventory Technical Manual 6.1, Table 4-1 (p. 4-2).
MANUFACTURED_CLASS = 'RES2'

# The occupancy classes the sampler covers: the single-family, mobile home
# and multi-family classes of Hazus Inventory Technical Manual 6.1, Table
# 4-1 (p. 4-2), whose dwellings the B25127 housing units describe. The
# table's other residential classes, RES4 to RES6, are not covered.
RESIDENTIAL_CLASSES = (
    'RES1',
    'RES2',
    'RES3A',
    'RES3B',
    'RES3C',
    'RES3D',
    'RES3E',
    'RES3F',
)

# B25127's cells: (year bin of COARSE_YEARS, class of SIZE_CLASSES,
# (owner-occupied variable, renter-occupied variable)). The two tenures
# are summed.
B25127_CELLS = (
    ('1939 or earlier', '1 unit', ('B25127_039E', 'B25127_082E')),
    ('1940 to 1959', '1 unit', ('B25127_032E', 'B25127_075E')),
    ('1960 to 1979', '1 unit', ('B25127_025E', 'B25127_068E')),
    ('1980 to 1999', '1 unit', ('B25127_018E', 'B25127_061E')),
    ('2000 to 2019', '1 unit', ('B25127_011E', 'B25127_054E')),
    ('2020 or later', '1 unit', ('B25127_004E', 'B25127_047E')),
    ('1939 or earlier', '2 to 4', ('B25127_040E', 'B25127_083E')),
    ('1940 to 1959', '2 to 4', ('B25127_033E', 'B25127_076E')),
    ('1960 to 1979', '2 to 4', ('B25127_026E', 'B25127_069E')),
    ('1980 to 1999', '2 to 4', ('B25127_019E', 'B25127_062E')),
    ('2000 to 2019', '2 to 4', ('B25127_012E', 'B25127_055E')),
    ('2020 or later', '2 to 4', ('B25127_005E', 'B25127_048E')),
    ('1939 or earlier', '5 to 19', ('B25127_041E', 'B25127_084E')),
    ('1940 to 1959', '5 to 19', ('B25127_034E', 'B25127_077E')),
    ('1960 to 1979', '5 to 19', ('B25127_027E', 'B25127_070E')),
    ('1980 to 1999', '5 to 19', ('B25127_020E', 'B25127_063E')),
    ('2000 to 2019', '5 to 19', ('B25127_013E', 'B25127_056E')),
    ('2020 or later', '5 to 19', ('B25127_006E', 'B25127_049E')),
    ('1939 or earlier', '20 to 49', ('B25127_042E', 'B25127_085E')),
    ('1940 to 1959', '20 to 49', ('B25127_035E', 'B25127_078E')),
    ('1960 to 1979', '20 to 49', ('B25127_028E', 'B25127_071E')),
    ('1980 to 1999', '20 to 49', ('B25127_021E', 'B25127_064E')),
    ('2000 to 2019', '20 to 49', ('B25127_014E', 'B25127_057E')),
    ('2020 or later', '20 to 49', ('B25127_007E', 'B25127_050E')),
    ('1939 or earlier', '50 or more', ('B25127_043E', 'B25127_086E')),
    ('1940 to 1959', '50 or more', ('B25127_036E', 'B25127_079E')),
    ('1960 to 1979', '50 or more', ('B25127_029E', 'B25127_072E')),
    ('1980 to 1999', '50 or more', ('B25127_022E', 'B25127_065E')),
    ('2000 to 2019', '50 or more', ('B25127_015E', 'B25127_058E')),
    ('2020 or later', '50 or more', ('B25127_008E', 'B25127_051E')),
    ('1939 or earlier', 'Manufactured home', ('B25127_044E', 'B25127_087E')),
    ('1940 to 1959', 'Manufactured home', ('B25127_037E', 'B25127_080E')),
    ('1960 to 1979', 'Manufactured home', ('B25127_030E', 'B25127_073E')),
    ('1980 to 1999', 'Manufactured home', ('B25127_023E', 'B25127_066E')),
    ('2000 to 2019', 'Manufactured home', ('B25127_016E', 'B25127_059E')),
    ('2020 or later', 'Manufactured home', ('B25127_009E', 'B25127_052E')),
)

# Part of the reproducibility contract; see keyed_uniforms.
ACS_YEAR_PURPOSE = 'year_built_b25034'

# Pseudo-units of the county-pooled conditional distribution added to each
# class row of a tract's B25127 counts; see the README beside this module,
# "Defaults and scope".
DEFAULT_SMOOTHING = 20.0

# The raking stops when the largest bin residual, as a share of the
# block group's fitted units, falls below RAKE_TOLERANCE, or after
# RAKE_ITERATIONS passes.
RAKE_TOLERANCE = 1e-12
RAKE_ITERATIONS = 10000

# The county prefix of a block-group or tract GEOID (state and county FIPS
# codes).
COUNTY_DIGITS = 5

# Smallest unit count an asset may carry.
MIN_UNITS = 1

# Width of a generated seed, matching the key width of the sampling
# helpers.
_SEED_BITS = 64

# Origins reported in fit_report.
SHARES_FROM_BLOCK_GROUP = 'block_group'
SHARES_FROM_TRACT = 'tract'
CONDITIONAL_FROM_TRACT = 'tract'
CONDITIONAL_FROM_COUNTY_POOL = 'county_pool'

FIT_REPORT_COLUMNS = (
    'shares_origin',
    'conditional_origin',
    'fitted_assets',
    'fitted_units',
    'negative_values',
    'iterations',
    'residual',
    'converged',
)

_BIN_LABELS = tuple(row[0] for row in B25034_BINS)
_BIN_VARIABLES = tuple(row[3] for row in B25034_BINS)
_FIRST_YEARS = np.array([row[1] for row in B25034_BINS], dtype=np.int64)
_LAST_YEARS = np.array([row[2] for row in B25034_BINS], dtype=np.int64)
_YEARS_PER_BIN = (_LAST_YEARS - _FIRST_YEARS + 1).astype(float)
_COARSE_LABELS = tuple(row[0] for row in COARSE_YEARS)
_COARSE_OF_BIN = np.array(
    [
        next(j for j, (_, members) in enumerate(COARSE_YEARS) if label in members)
        for label in _BIN_LABELS
    ],
    dtype=np.int64,
)
_B25127_VARIABLES = tuple(
    variable for _, _, pair in B25127_CELLS for variable in pair
)
_N_BINS = len(B25034_BINS)
_N_COARSE = len(COARSE_YEARS)
_N_CLASSES = len(SIZE_CLASSES)


def size_class(occupancy: str, units: float) -> int:
    """
    Place one residential asset in a class of :data:`SIZE_CLASSES`.

    RES2 falls in the manufactured-home class whatever its count. Any other
    class is placed by its unit count against :data:`SIZE_CLASS_BOUNDS`,
    inclusive above. The count is not validated.

    Args:
        occupancy (str):
            The asset's occupancy class.
        units (float):
            The asset's unit count, possibly fractional.

    Returns:
        int:
            The index of the class in :data:`SIZE_CLASSES`.
    """
    if occupancy == MANUFACTURED_CLASS:
        return SIZE_CLASSES.index('Manufactured home')
    for index, bound in enumerate(SIZE_CLASS_BOUNDS):
        if units <= bound:
            return index
    return len(SIZE_CLASS_BOUNDS)


def _is_geoid(value: object, digits: int) -> bool:
    """
    Say whether a value is a GEOID string of a given length.

    Args:
        value (object):
            The value to test.
        digits (int):
            The required number of digits.

    Returns:
        bool:
            ``True`` for a string of exactly ``digits`` ASCII digits.
    """
    return (
        isinstance(value, str)
        and len(value) == digits
        and value.isascii()
        and value.isdigit()
    )


def _table_arrays(
    table: pd.DataFrame, columns: tuple[str, ...], digits: int, name: str
) -> dict[str, np.ndarray]:
    """
    Read the required columns of a Census table as one array per GEOID.

    Args:
        table (pd.DataFrame):
            The table, indexed by GEOID string.
        columns (tuple[str, ...]):
            The required variable ids, in the order of the arrays.
        digits (int):
            The length of every GEOID.
        name (str):
            The argument name, for error messages.

    Returns:
        dict[str, np.ndarray]:
            GEOID to the values of the required columns, as floats.

    Raises:
        ValueError:
            If a required column is missing, a value is not a finite
            number, or an index entry is not a GEOID of ``digits``
            digits or occurs twice.
    """
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(
            f'{name} lacks required columns ({len(missing)}), such as '
            f'{missing[0]!r}.'
        )
    bad_ids = [geoid for geoid in table.index if not _is_geoid(geoid, digits)]
    if bad_ids:
        raise ValueError(
            f'{name} must be indexed by GEOID strings of {digits} digits. '
            f'Index entries that are not: {len(bad_ids)}, such as '
            f'{bad_ids[0]!r}.'
        )
    if table.index.has_duplicates:
        repeated = table.index[table.index.duplicated()][0]
        raise ValueError(f'{name} lists GEOID {repeated!r} more than once.')
    try:
        values = table.loc[:, list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'{name} holds a value in a required column that is not a number.'
        ) from error
    if not np.isfinite(values).all():
        raise ValueError(
            f'{name} holds a value in a required column that is not finite.'
        )
    return dict(zip(table.index, values))


def _normalize_rows(joint: np.ndarray) -> np.ndarray:
    """
    Normalize each row of a matrix to sum to one.

    Args:
        joint (np.ndarray):
            Non-negative counts.

    Returns:
        np.ndarray:
            The rows divided by their sums. A row summing to zero becomes
            uniform.
    """
    totals = joint.sum(axis=1, keepdims=True)
    out = np.divide(joint, np.where(totals > 0, totals, 1.0))
    out[totals.ravel() <= 0] = 1.0 / joint.shape[1]
    return out


def _rake(
    seed: np.ndarray, class_units: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, int, float]:
    """
    Rake class rows so that the expected units per year bin meet a target.

    Iterative proportional fitting on the logarithm of the weights: each
    row is renormalized to one, and each bin's multiplier is updated
    until the largest bin residual falls below :data:`RAKE_TOLERANCE`
    or :data:`RAKE_ITERATIONS` passes are spent.

    Args:
        seed (np.ndarray):
            Classes by year bins, non-negative starting weights, each row
            holding some mass on a bin whose target is positive.
        class_units (np.ndarray):
            The units of each class, the row weights.
        target (np.ndarray):
            The units wanted per year bin, summing to the class units.

    Returns:
        tuple[np.ndarray, int, float]:
            The raked rows (each summing to one), the passes used, and
            the final residual as a share of the units.
    """
    active = target > 0
    with np.errstate(divide='ignore'):
        log_seed = np.log(seed)
    total = math.fsum(class_units)
    log_multiplier = np.zeros(seed.shape[1])
    probabilities = seed
    residual = math.inf
    passes = 0
    for passes in range(1, RAKE_ITERATIONS + 1):  # noqa: B007
        weights = log_seed + log_multiplier
        weights = weights - np.max(weights, axis=1, keepdims=True)
        probabilities = np.exp(weights)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        column = class_units @ probabilities
        residual = float(np.abs(column - target).max() / total)
        if residual < RAKE_TOLERANCE:
            break
        movable = active & (column > 0)
        log_multiplier[movable] += np.log(target[movable] / column[movable])
    return probabilities, passes, residual


class ACSYearBuiltSampler(InferenceEngine):
    """
    Draw construction years for residential assets from Census tables.

    :meth:`fit` fits one distribution over the year bins of
    :data:`B25034_BINS` per block group and class of
    :data:`SIZE_CLASSES`, from the block group's B25034 counts and its
    tract's B25127 counts, and :meth:`draw` draws each asset's year from
    its distribution. Where neither a block group nor its tract holds
    B25034 units, or the B25127 table holds no tract of its county, the
    block group stays unfitted and :meth:`check` refuses its assets. The
    README beside this module describes the method and the defaults.

    :meth:`fit` reads every asset of the inventory it is given, and a draw
    depends on the other residential assets of its block group and on
    every tract of its county in the B25127 table. A call of :meth:`check`
    or :meth:`draw` on a one-asset inventory reads that asset's
    distribution in the fitted inventory.

    The sampler covers :data:`RESIDENTIAL_CLASSES`. It writes
    ``year_built_key`` whether or not one is already there; keeping an
    observed year is the caller's decision.
    """

    def __init__(
        self,
        b25034_block_groups: pd.DataFrame,
        b25034_tracts: pd.DataFrame,
        b25127_tracts: pd.DataFrame,
        *,
        seed: int | None = None,
        world: int = DEFAULT_WORLD,
        smoothing: float = DEFAULT_SMOOTHING,
        block_group_key: str = BLOCK_GROUP_KEY,
        occupancy_key: str = 'OccupancyClass',
        units_key: str = 'NumberOfUnits',
        year_built_key: str = 'YearBuilt',
    ) -> None:
        """
        Read the Census tables and validate the configuration.

        Args:
            b25034_block_groups (pd.DataFrame):
                Table B25034 per block group, indexed by GEOID string of
                :data:`~brails.constants.GEOID_LENGTH` digits, with the
                estimate variables of :data:`B25034_BINS` as columns.
                Other columns are ignored. A negative value, the Census
                placeholder for an unavailable estimate, reads as zero.
            b25034_tracts (pd.DataFrame):
                Table B25034 per tract, indexed by GEOID string of
                :data:`~brails.constants.TRACT_GEOID_LENGTH` digits, in the
                same form.
            b25127_tracts (pd.DataFrame):
                Table B25127 per tract, indexed by GEOID string of
                :data:`~brails.constants.TRACT_GEOID_LENGTH` digits, with the
                variables of :data:`B25127_CELLS` as columns, in the same form. Each
                county's pool sums every tract of this table in that
                county, and a tract's conditional distribution is smoothed
                toward it.
            seed (int | None):
                Seed for the draw. If None, a seed is drawn from system
                entropy and recorded on ``self.seed``.
            world (int):
                Possible-world index. A world other than the default
                requires an explicit seed.
            smoothing (float):
                Pseudo-units of the county-pooled conditional added to
                each class row of a tract's B25127 counts.
            block_group_key (str):
                Feature holding the asset's block-group GEOID.
            occupancy_key (str):
                Feature holding the occupancy class.
            units_key (str):
                Feature holding the unit count.
            year_built_key (str):
                Feature name the drawn year is written to.

        Raises:
            ValueError:
                If a table lacks a required column, holds a value in one
                that is not a finite number, or is not indexed by unique
                GEOID strings of its level's length; if ``smoothing`` is
                negative or not finite; or if ``world`` is not the default
                and no seed is given.
        """
        if not (math.isfinite(smoothing) and smoothing >= 0):
            raise ValueError(
                f'smoothing must be a finite number of at least 0, got '
                f'{smoothing!r}.'
            )
        if int(world) != DEFAULT_WORLD and seed is None:
            raise ValueError(
                f'World {world} needs the seed of the run; samplers of '
                'one run share one seed.'
            )
        self._b25034_block_groups = _table_arrays(
            b25034_block_groups,
            _BIN_VARIABLES,
            GEOID_LENGTH,
            'b25034_block_groups',
        )
        self._b25034_tracts = _table_arrays(
            b25034_tracts, _BIN_VARIABLES, TRACT_GEOID_LENGTH, 'b25034_tracts'
        )
        b25127 = _table_arrays(
            b25127_tracts, _B25127_VARIABLES, TRACT_GEOID_LENGTH, 'b25127_tracts'
        )
        # per tract: B25127 counts by size class and year bin, both tenures
        # summed and negative values read as zero, with the negatives counted
        self._b25127_joint = {}
        for tract, values in b25127.items():
            joint = np.zeros((_N_CLASSES, _N_COARSE))
            clean = np.where(values < 0, 0.0, values)
            for k, (year_bin, size, _) in enumerate(B25127_CELLS):
                row = SIZE_CLASSES.index(size)
                column = _COARSE_LABELS.index(year_bin)
                joint[row, column] = math.fsum(clean[2 * k : 2 * k + 2])
            self._b25127_joint[tract] = (joint, int((values < 0).sum()))
        # exact (fsum) sums of the county's tract tables, each class row
        # normalized to one
        by_county: dict[str, list[np.ndarray]] = {}
        for tract, (counts, _) in self._b25127_joint.items():
            by_county.setdefault(tract[:COUNTY_DIGITS], []).append(counts)
        self._county_pools = {}
        for county, rows in by_county.items():
            flat = np.stack(rows).reshape(len(rows), -1)
            sums = [math.fsum(flat[:, k]) for k in range(flat.shape[1])]
            self._county_pools[county] = _normalize_rows(
                np.array(sums, dtype=float).reshape(rows[0].shape)
            )
        self.seed = secrets.randbits(_SEED_BITS) if seed is None else int(seed)
        self.world = int(world)
        self.smoothing = float(smoothing)
        self.block_group_key = block_group_key
        self.occupancy_key = occupancy_key
        self.units_key = units_key
        self.year_built_key = year_built_key
        self._vectors: dict[str, np.ndarray] | None = None
        self._unfitted: dict[str, str] = {}
        self._report: pd.DataFrame | None = None

    # -----------------------------------------------------------------
    # Census tables

    def _bin_shares(
        self, block_group: str
    ) -> tuple[np.ndarray | None, str | None, int]:
        """
        Read a block group's B25034 bin shares, or its tract's.

        Args:
            block_group (str):
                The block-group GEOID.

        Returns:
            tuple[np.ndarray | None, str | None, int]:
                The bin shares, their origin, and the count of negative
                values read as zero. The shares and the origin are
                ``None`` where neither the block group nor its tract holds
                units.
        """
        negatives = 0
        for origin, table, geoid in (
            (SHARES_FROM_BLOCK_GROUP, self._b25034_block_groups, block_group),
            (
                SHARES_FROM_TRACT,
                self._b25034_tracts,
                block_group[:TRACT_GEOID_LENGTH],
            ),
        ):
            values = table.get(geoid)
            if values is None:
                continue
            negatives += int((values < 0).sum())
            counts = np.where(values < 0, 0.0, values)
            total = math.fsum(counts)
            if total > 0:
                return counts / total, origin, negatives
        return None, None, negatives

    def _conditional(
        self, tract: str
    ) -> tuple[np.ndarray | None, str | None, int]:
        """
        Build a tract's smoothed B25127 conditional.

        Args:
            tract (str):
                The tract GEOID.

        Returns:
            tuple[np.ndarray | None, str | None, int]:
                The probability of each year bin given each class, one
                row per class of :data:`SIZE_CLASSES`, its origin, and the
                negative values read as zero. The conditional and origin
                are ``None`` where the B25127 table holds no tract of the
                county.
        """
        pool = self._county_pools.get(tract[:COUNTY_DIGITS])
        if pool is None:
            return None, None, 0
        counts, negatives = self._b25127_joint.get(tract, (None, 0))
        if counts is None:
            return pool.copy(), CONDITIONAL_FROM_COUNTY_POOL, 0
        smoothed = counts + self.smoothing * pool
        return _normalize_rows(smoothed), CONDITIONAL_FROM_TRACT, negatives

    def _missing_table(self, block_group: str) -> str | None:
        """
        Say which table keeps a block group from being fitted.

        Args:
            block_group (str):
                The block-group GEOID.

        Returns:
            str | None:
                The reason, or ``None`` where both tables cover it.
        """
        shares, _, _ = self._bin_shares(block_group)
        if shares is None:
            return (
                f'table B25034 holds no units for block group {block_group} '
                f'or its tract {block_group[:TRACT_GEOID_LENGTH]}'
            )
        if block_group[:COUNTY_DIGITS] not in self._county_pools:
            return (
                'table B25127 holds no tract of county '
                f'{block_group[:COUNTY_DIGITS]}'
            )
        return None

    # -----------------------------------------------------------------
    # Fit

    def fit(self, input_inventory: AssetInventory) -> ACSYearBuiltSampler:
        """
        Fit the year distributions of every block group in an inventory.

        The fit reads the occupancy, units and block group of every asset
        and changes none. Assets with an absent or unusable occupancy, unit
        count or block group, or a class the sampler does not cover, do
        not enter it. An asset holding one value per world in any of the
        three raises. A later call replaces the earlier fit.

        Args:
            input_inventory (AssetInventory):
                One world's scalar inventory, holding every residential
                asset of the study area.

        Returns:
            ACSYearBuiltSampler:
                This sampler, fitted.

        Raises:
            ValueError:
                If an asset holds one value per world in the occupancy,
                units or block-group feature.
        """
        units: dict[str, list[list[float]]] = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            diagnosis, detail, _ = self._resolve(
                features, input_inventory.vector_features, fitted=False
            )
            if diagnosis == DIAGNOSIS_UNSUPPORTED:
                raise ValueError(
                    f'Asset {asset_id}: {detail}. fit takes one world at a '
                    'time.'
                )
            if diagnosis != DIAGNOSIS_OK:
                continue
            occupancy = features[self.occupancy_key]
            count = float(features[self.units_key])
            block_group = features[self.block_group_key]
            per_class = units.setdefault(
                block_group, [[] for _ in range(_N_CLASSES)]
            )
            per_class[size_class(occupancy, count)].append(count)

        vectors: dict[str, np.ndarray] = {}
        unfitted: dict[str, str] = {}
        report = {}
        for block_group in sorted(units):
            per_class = units[block_group]
            class_units = np.array([math.fsum(values) for values in per_class])
            n_assets = sum(len(values) for values in per_class)
            fitted, row, reason = self._fit_block_group(block_group, class_units)
            row['fitted_assets'] = n_assets
            report[block_group] = row
            if fitted is None:
                unfitted[block_group] = reason
            else:
                vectors[block_group] = fitted
        self._vectors = vectors
        self._unfitted = unfitted
        frame = pd.DataFrame.from_dict(
            report, orient='index', columns=list(FIT_REPORT_COLUMNS)
        )
        frame.index.name = self.block_group_key
        self._report = frame
        return self

    def _fit_block_group(
        self, block_group: str, class_units: np.ndarray
    ) -> tuple[np.ndarray | None, dict, str | None]:
        """
        Fit the class rows of one block group.

        Args:
            block_group (str):
                The block-group GEOID.
            class_units (np.ndarray):
                The summed units of the block group's fitted assets in
                each class of :data:`SIZE_CLASSES`.

        Returns:
            tuple[np.ndarray | None, dict, str | None]:
                The raked rows, one per class over the year bins, or
                ``None`` where a table is missing; the block group's
                report row; and the reason it was not fitted, if it was
                not.
        """
        fitted_units = math.fsum(class_units)
        row = {
            'shares_origin': None,
            'conditional_origin': None,
            'fitted_units': fitted_units,
            'negative_values': 0,
            'iterations': 0,
            'residual': math.nan,
            'converged': False,
        }
        shares, shares_origin, negatives = self._bin_shares(block_group)
        row['shares_origin'] = shares_origin
        row['negative_values'] = negatives
        reason = self._missing_table(block_group)
        if reason is not None:
            return None, row, reason
        conditional, conditional_origin, negatives = self._conditional(
            block_group[:TRACT_GEOID_LENGTH]
        )
        row['conditional_origin'] = conditional_origin
        row['negative_values'] += negatives

        if fitted_units <= 0:
            row['residual'] = 0.0
            row['converged'] = True
            return np.tile(shares, (_N_CLASSES, 1)), row, None

        within = np.zeros(_N_BINS)
        for j in range(_N_COARSE):
            members = np.equal(_COARSE_OF_BIN, j)
            total = math.fsum(shares[members])
            within[members] = (
                shares[members] / total if total > 0 else 1.0 / members.sum()
            )
        seed = conditional[:, _COARSE_OF_BIN] * within
        seed[:, shares <= 0] = 0.0
        dead = seed.sum(axis=1) <= 0
        seed[dead] = shares
        raked, passes, residual = _rake(seed, class_units, shares * fitted_units)
        row['iterations'] = passes
        row['residual'] = residual
        row['converged'] = residual < RAKE_TOLERANCE
        return raked, row, None

    @property
    def vectors(self) -> pd.DataFrame:
        """
        The fitted distributions, one row per block group and class.

        Returns:
            pd.DataFrame:
                Indexed by block group and class of :data:`SIZE_CLASSES`,
                with the bin labels of :data:`B25034_BINS` as
                columns, each row summing to one. A copy.

        Raises:
            RuntimeError:
                If the sampler has not been fitted.
        """
        fitted = self._require_fit()
        index = pd.MultiIndex.from_product(
            [sorted(fitted), SIZE_CLASSES],
            names=[self.block_group_key, 'size_class'],
        )
        values = (
            np.vstack([fitted[bg] for bg in sorted(fitted)])
            if fitted
            else np.zeros((0, _N_BINS))
        )
        return pd.DataFrame(values, index=index, columns=list(_BIN_LABELS))

    @property
    def fit_report(self) -> pd.DataFrame:
        """
        How each block group among the fitted assets was fitted.

        Returns:
            pd.DataFrame:
                One row per block group, with the origin of the shares
                (:data:`SHARES_FROM_BLOCK_GROUP` or
                :data:`SHARES_FROM_TRACT`) and of the conditional
                distribution (:data:`CONDITIONAL_FROM_TRACT` or
                :data:`CONDITIONAL_FROM_COUNTY_POOL`), each
                missing (NaN under pandas 3) where the table is missing;
                the fitted assets and units; the negative table values
                read as zero for the block group's and tract's rows; the
                raking passes; the final residual as a share of the units;
                and whether it met :data:`RAKE_TOLERANCE`. A block group
                that missed the tolerance keeps its last rows. A copy.

        Raises:
            RuntimeError:
                If the sampler has not been fitted.
        """
        self._require_fit()
        return self._report.copy()

    def _require_fit(self) -> dict[str, np.ndarray]:
        """
        Return the fitted rows, or raise if there are none.

        Returns:
            dict[str, np.ndarray]:
                Block-group GEOID to its raked rows.

        Raises:
            RuntimeError:
                If :meth:`fit` has not been called.
        """
        if self._vectors is None:
            raise RuntimeError(
                'ACSYearBuiltSampler must be fitted with fit() first.'
            )
        return self._vectors

    # -----------------------------------------------------------------
    # Check

    def check(self, input_inventory: AssetInventory) -> pd.DataFrame:
        """
        Diagnose every asset without drawing anything.

        Args:
            input_inventory (AssetInventory):
                The inventory to diagnose.

        Returns:
            pd.DataFrame:
                One row per asset id, with ``diagnosis``, ``detail`` and
                ``resolvable_by``. A driver holding one value per world is
                unsupported. An absent or unusable occupancy, unit count or
                block group is invalid. A class the sampler does not cover,
                or a block group without a fitted distribution, is
                infeasible. ``resolvable_by`` names the failing driver
                where it is the only one, since changing it alone would
                admit the asset, and names nothing where several fail.

        Raises:
            RuntimeError:
                If the sampler has not been fitted.
        """
        self._require_fit()
        rows = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            rows[asset_id] = self._resolve(
                features, input_inventory.vector_features
            )
        return diagnosis_frame(rows)

    def _resolve(
        self, features: dict, vector_features: set, *, fitted: bool = True
    ) -> tuple[str, str, tuple[str, ...]]:
        """
        Diagnose one asset from its features and the fitted table.

        Args:
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors.
            fitted (bool):
                Whether to require a fitted distribution for the asset's
                block group. :meth:`fit` passes ``False``.

        Returns:
            tuple[str, str, tuple[str, ...]]:
                The diagnosis, the detail, and the features that would
                each, alone, admit the asset.
        """
        keys = (self.occupancy_key, self.units_key, self.block_group_key)
        for key in keys:
            if is_probabilistic_feature(key, features.get(key), vector_features):
                return (
                    DIAGNOSIS_UNSUPPORTED,
                    f'{key} holds one value per world; the sampler takes '
                    'one world at a time',
                    (),
                )
        invalid: list[tuple[str, str]] = []
        uncovered: list[tuple[str, str]] = []
        for key, (bad, outside) in (
            (self.occupancy_key, self._occupancy_failure(features)),
            (
                self.units_key,
                (self._units_failure(features.get(self.units_key)), None),
            ),
            (
                self.block_group_key,
                self._block_group_failure(features, fitted=fitted),
            ),
        ):
            if bad is not None:
                invalid.append((key, bad))
            if outside is not None:
                uncovered.append((key, outside))
        failing = invalid + uncovered
        if not failing:
            return DIAGNOSIS_OK, '', ()
        details = '; '.join(message for _, message in failing)
        resolvable = (failing[0][0],) if len(failing) == 1 else ()
        if invalid:
            return DIAGNOSIS_INVALID, details, resolvable
        return DIAGNOSIS_INFEASIBLE, details, resolvable

    def _occupancy_failure(self, features: dict) -> tuple[str | None, str | None]:
        """
        Say why an occupancy class cannot be used.

        Args:
            features (dict):
                The asset's features.

        Returns:
            tuple[str | None, str | None]:
                The reason the value is absent or unusable, and the reason
                it is a class the sampler does not cover; each ``None``
                where it does not apply.
        """
        occupancy = features.get(self.occupancy_key)
        if not isinstance(occupancy, str):
            return (
                f'{self.occupancy_key} is absent or not a string: '
                f'{occupancy!r}',
                None,
            )
        if occupancy not in RESIDENTIAL_CLASSES:
            return (
                None,
                f'{self.occupancy_key} {occupancy!r} is outside the '
                'residential classes the sampler covers',
            )
        return None, None

    def _block_group_failure(
        self, features: dict, *, fitted: bool
    ) -> tuple[str | None, str | None]:
        """
        Say why a block group cannot be used.

        Args:
            features (dict):
                The asset's features.
            fitted (bool):
                Whether to require a fitted distribution for it.

        Returns:
            tuple[str | None, str | None]:
                The reason the value is absent or not a GEOID, and the
                reason it has no fitted distribution; each ``None`` where
                it does not apply.
        """
        block_group = features.get(self.block_group_key)
        if not _is_geoid(block_group, GEOID_LENGTH):
            return (
                f'{self.block_group_key} is absent or not a GEOID string '
                f'of {GEOID_LENGTH} digits: {block_group!r}',
                None,
            )
        if not fitted or block_group in self._vectors:
            return None, None
        reason = self._unfitted.get(block_group) or self._missing_table(
            block_group
        )
        if reason is None:
            reason = f'block group {block_group} had no asset in the fit'
        return (
            None,
            f'{self.block_group_key} {block_group} is not fitted: {reason}',
        )

    def _units_failure(self, units: object) -> str | None:
        """
        Say why a unit count cannot be used.

        Args:
            units (object):
                The value read from the asset.

        Returns:
            str | None:
                The reason the value is unusable, or ``None`` where it is
                usable.
        """
        if units is None:
            return f'no {self.units_key}'
        if isinstance(units, bool):
            return f'{self.units_key} is a boolean: {units!r}'
        try:
            count = float(units)
        except (TypeError, ValueError, OverflowError):
            return f'{self.units_key} is not a number: {units!r}'
        if not math.isfinite(count):
            return f'{self.units_key} is not finite: {units!r}'
        if count < MIN_UNITS:
            return f'{self.units_key} of {units!r} is below {MIN_UNITS}'
        return None

    # -----------------------------------------------------------------
    # Draw

    def draw(
        self,
        input_inventory: AssetInventory,
        *,
        iteration: int = DEFAULT_ITERATION,
    ) -> pd.Series:
        """
        Draw a year for every asset. Every asset must pass check.

        Each asset inverts one keyed uniform through its fitted
        distribution spread evenly over the years of each bin, so one
        uniform picks both the bin and the year.

        Args:
            input_inventory (AssetInventory):
                An inventory whose assets all pass :meth:`check`.
            iteration (int):
                Resolution-loop attempt number. Distinct iterations draw
                independently for the same asset.

        Returns:
            pd.Series:
                Integer years indexed by asset id, named
                ``year_built_key``.

        Raises:
            RuntimeError:
                If the sampler has not been fitted.
            ValueError:
                If an asset does not pass :meth:`check`, naming the
                assets.
        """
        fitted = self._require_fit()
        asset_ids = []
        rows = []
        failures = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            diagnosis, detail, _ = self._resolve(
                features, input_inventory.vector_features
            )
            if diagnosis != DIAGNOSIS_OK:
                failures[asset_id] = detail
                continue
            asset_ids.append(asset_id)
            rows.append(
                (
                    features[self.block_group_key],
                    size_class(
                        features[self.occupancy_key],
                        float(features[self.units_key]),
                    ),
                )
            )
        if failures:
            raise ValueError(
                f'draw needs assets that pass check(); {len(failures)} '
                f'fail: {preview_failures(pd.Series(failures, dtype=object))}.'
            )
        if not asset_ids:
            return pd.Series(dtype='int64', name=self.year_built_key)
        distributions = sorted(set(rows))
        codes = {row: code for code, row in enumerate(distributions)}
        # spread each bin's weight evenly over its years
        weights = [
            np.repeat(
                fitted[block_group][index] / _YEARS_PER_BIN,
                _YEARS_PER_BIN.astype(int),
            )
            for block_group, index in distributions
        ]
        uniforms = keyed_uniforms(
            asset_key_digest(asset_ids),
            seed=self.seed,
            purpose=ACS_YEAR_PURPOSE,
            world=self.world,
            iteration=iteration,
        )
        positions = grouped_categorical(
            uniforms,
            weights=weights,
            groups=np.array([codes[row] for row in rows], dtype=np.int64),
        )
        return pd.Series(
            OLDEST_YEAR + positions,
            index=pd.Index(asset_ids, dtype=object),
            name=self.year_built_key,
        )

    def infer(self, input_inventory: AssetInventory) -> AssetInventory:
        """
        Draw a construction year for every asset in the inventory.

        Args:
            input_inventory (AssetInventory):
                The inventory to process. Every asset must pass
                :meth:`check`; filter the inventory with it first if that
                is not guaranteed.

        Returns:
            AssetInventory:
                A copy of the input with the drawn year written under
                ``year_built_key``, replacing any year already there.

        Raises:
            RuntimeError:
                If the sampler has not been fitted.
            ValueError:
                If any asset cannot be drawn for, naming the assets and
                the reason each fails.
        """
        verdicts = self.check(input_inventory)
        failed = verdicts.loc[verdicts['diagnosis'] != DIAGNOSIS_OK, 'detail']
        if not failed.empty:
            raise ValueError(
                f'Cannot sample {self.year_built_key} for {len(failed)} of '
                f'{len(verdicts)} assets: {preview_failures(failed)}. Remove '
                'them or correct their data, then call infer() again.'
            )
        years = self.draw(input_inventory)
        output_inventory = deepcopy(input_inventory)
        for asset_id, year in years.items():
            output_inventory.add_asset_features(
                asset_id, {self.year_built_key: int(year)}
            )
        return output_inventory
