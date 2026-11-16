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
Assign Hazus seismic design levels from construction years.

.. autosummary::

      HazusEQDesignLevelInferer
      assign_design_levels
      apply_w1_carveout
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    get_hazus_state_region_mapping,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
    diagnosis_frame,
    preview_failures,
    preview_ids,
)
from brails.inferers.inference_engine import InferenceEngine
from brails.types.asset_inventory import is_probabilistic_feature

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory

# Design levels by construction era for California, from the FEMA Hazus
# Inventory Technical Manual 6.1, Appendix C, Table C-10. Each zone row
# pairs the first years of the later eras with one level per era. The
# table runs through 1996; the v5.1 method carries its last era forward.
# The year-to-edition ladder of Table C-9 belongs to the v6.1 method.
CA_DESIGN_LEVELS = {
    '4': ((1941, 1976), ('Pre-Code', 'Moderate-Code', 'High-Code')),
    '3': ((1941, 1976), ('Pre-Code', 'Moderate-Code', 'Moderate-Code')),
}

# Design levels by construction era for the general building stock in
# states without their own matrix, from the FEMA Hazus Inventory
# Technical Manual 6.1, Table 5-12 (numbered 5-13 in the List of Tables,
# p. 5-16), keyed by 1997 UBC seismic zone. The source bounds the last
# era at the first IBC/IRC adoption; the v5.1 method does not consult
# adoption years.
GENERIC_DESIGN_LEVELS = {
    '4': ((1976, 1996), ('Pre-Code', 'Moderate-Code', 'High-Code')),
    '3': ((1976, 1996), ('Pre-Code', 'Moderate-Code', 'Moderate-Code')),
    '2B': ((1976, 1996), ('Pre-Code', 'Low-Code', 'Moderate-Code')),
    '2A': ((1976, 1996), ('Pre-Code', 'Low-Code', 'Low-Code')),
    '1': ((1976, 1996), ('Pre-Code', 'Low-Code', 'Low-Code')),
    '0': ((1976, 1996), ('Pre-Code', 'Pre-Code', 'Low-Code')),
}

# The largest magnitude the era comparison can represent. Era selection
# casts to a 64-bit integer, and a value further from zero than this
# casts to something else entirely, which would assign a design level to
# a year nobody supplied.
MAX_REPRESENTABLE_YEAR = int(np.iinfo(np.int64).max)

# States with their own Appendix C matrix, keyed by the full state name
# that Table A-22 uses.
STATE_DESIGN_LEVELS = {'California': CA_DESIGN_LEVELS}


def _outside_year_bounds(value: object) -> bool:
    """
    Say whether a year sits outside the range the era comparison reads.

    Args:
        value (object):
            A raw year value, of any type.

    Returns:
        bool:
            True where the value sits further from zero than
            :data:`MAX_REPRESENTABLE_YEAR`, in either direction. A value
            that reads as no number at all gives False.
    """
    if isinstance(value, bool):
        return False
    try:
        # An integer is compared as an integer. float(2**63 - 1) rounds up
        # to 2**63, which an exact comparison against the bound refuses.
        return abs(int(value)) > MAX_REPRESENTABLE_YEAR
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(number) and abs(number) > MAX_REPRESENTABLE_YEAR)


def _select_era_row(
    state: str, seismic_zone: str | int
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """
    Pick the era breaks and design levels for a state and seismic zone.

    Args:
        state (str):
            Full state name, as in ``'California'``.
        seismic_zone (str | int):
            1997 UBC seismic zone.

    Returns:
        tuple[tuple[int, ...], tuple[str, ...]]:
            The matching row of the design level table.

    Raises:
        ValueError:
            If the state is not in the Hazus region table, or the zone is
            not in the selected state table.
    """
    if state not in get_hazus_state_region_mapping():
        raise ValueError(
            f'{state!r} is not a state in the Hazus region table. Give '
            "the full name, as in 'California'; two-letter codes and "
            'abbreviations are not accepted.'
        )
    table = STATE_DESIGN_LEVELS.get(state, GENERIC_DESIGN_LEVELS)
    zone = str(seismic_zone).upper()
    if zone not in table:
        valid = ', '.join(table)
        raise ValueError(
            f'Seismic zone {seismic_zone!r} is not in the design level '
            f'table selected by state {state!r}. Valid zones: {valid}.'
        )
    return table[zone]


def assign_design_levels(
    year_built: pd.Series,
    state: str = 'California',
    seismic_zone: str | int = '4',
    *,
    field_name: str = 'YearBuilt',
) -> pd.Series:
    """
    Assign Hazus v5.1 seismic design levels from construction years.

    California uses its Appendix C matrix (Table C-10). Other states
    fall back to the generic zone table (Table 5-12). Years before the
    first era break map to the first level, and the last era of each
    table continues to the present.

    Args:
        year_built (pd.Series):
            Construction years indexed by asset id.
        state (str):
            Full state name that selects the matrix, as in
            ``'California'``.
        seismic_zone (str | int):
            1997 UBC seismic zone of the site.
        field_name (str):
            Name of the year feature, used in error messages.

    Returns:
        pd.Series:
            The design level of each asset, on the input index, named
            ``DesignLevel``.

    Raises:
        ValueError:
            If the state is not in the Hazus region table, the zone is
            not in the selected state table, or a year is missing,
            non-numeric, non-integral, not positive, a boolean, or
            further from zero than :data:`MAX_REPRESENTABLE_YEAR`.
    """
    breaks, levels = _select_era_row(state, seismic_zone)
    # A Python integer too large in magnitude for a float raises out of the
    # coercion, so values outside the bounds are set aside before it runs.
    too_large = year_built.map(_outside_year_bounds)
    years = pd.to_numeric(year_built.mask(too_large), errors='coerce')
    # A boolean survives numeric coercion as 0 or 1 and would be dated
    # Pre-Code, so it is refused on its type.
    boolean = year_built.map(lambda value: isinstance(value, bool))
    invalid = years.isna() | (years <= 0) | (years % 1 != 0) | too_large | boolean
    if invalid.any():
        offenders = list(year_built.index[invalid])
        raise ValueError(
            f'{field_name} must be a positive whole year. Missing or '
            f'invalid values on {len(offenders)} of {len(year_built)} '
            f'assets: {preview_ids(offenders)}.'
        )
    era = np.searchsorted(breaks, years.to_numpy(dtype=int), side='right')
    assigned = np.asarray(levels, dtype=object)[era]
    return pd.Series(assigned, index=year_built.index, name='DesignLevel')


def apply_w1_carveout(
    design_levels: pd.Series,
    structure_types: pd.Series,
) -> pd.Series:
    """
    Raise Pre-Code W1 buildings to Moderate-Code.

    Table C-10 assumes Moderate-Code for W1 buildings of the Pre-Code
    era, an assumption that "will be implemented following assignment
    of earthquake specific building types". The function does not check
    the state; callers apply it only where the state's table has the
    W1 note.

    Args:
        design_levels (pd.Series):
            Design levels indexed by asset id.
        structure_types (pd.Series):
            Hazus structure types on the same index.

    Returns:
        pd.Series:
            A new series with the carve-out applied; the inputs are
            unchanged.

    Raises:
        ValueError:
            If the two series do not have the same index.
    """
    if not design_levels.index.equals(structure_types.index):
        raise ValueError(
            'design_levels and structure_types must have the same index.'
        )
    adjusted = design_levels.copy()
    is_pre_code_w1 = (design_levels == 'Pre-Code') & (structure_types == 'W1')
    adjusted[is_pre_code_w1] = 'Moderate-Code'
    return adjusted


class HazusEQDesignLevelInferer(InferenceEngine):
    """
    Add a Hazus design level to every asset in an inventory.

    An inventory wrapper around :func:`assign_design_levels`. Run
    :func:`apply_w1_carveout` afterwards, once structure types are
    drawn.

    :meth:`check` diagnoses every asset, :meth:`draw` assumes they all
    passed, and :meth:`infer` runs both and raises on any failure.
    """

    def __init__(
        self,
        method: str = '5.1',
        state: str = 'California',
        seismic_zone: str | int = '4',
        year_built_key: str = 'YearBuilt',
        design_level_key: str = 'DesignLevel',
    ) -> None:
        """
        Set up the inferer and validate its configuration.

        Args:
            method (str):
                The design level method. Only ``'5.1'`` is implemented.
            state (str):
                Full state name that selects the matrix, as in
                ``'California'``.
            seismic_zone (str | int):
                1997 UBC seismic zone of the site.
            year_built_key (str):
                Feature name the construction year is read from.
            design_level_key (str):
                Feature name the design level is written to.

        Raises:
            NotImplementedError:
                If ``method`` is ``'6.1'``.
            ValueError:
                If ``method`` is unknown, the state is not in the Hazus
                region table, or the zone is not in the selected state
                table.
        """
        if str(method) == '6.1':
            raise NotImplementedError(
                'The v6.1 design level method is not implemented yet.'
            )
        if str(method) != '5.1':
            raise ValueError(
                f'Unknown design level method {method!r}. Supported: 5.1.'
            )
        _select_era_row(state, seismic_zone)
        self.method = str(method)
        self.state = state
        self.seismic_zone = seismic_zone
        self.year_built_key = year_built_key
        self.design_level_key = design_level_key

    def check(self, input_inventory: AssetInventory) -> pd.DataFrame:
        """
        Diagnose every asset without assigning anything.

        Args:
            input_inventory (AssetInventory):
                The inventory to diagnose.

        Returns:
            pd.DataFrame:
                One row per asset id, with ``diagnosis``, ``detail`` and
                ``resolvable_by``. A year the table cannot read is
                invalid and names the year feature, which a caller may
                replace. A per-world list is unsupported and names
                nothing.
        """
        rows = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            rows[asset_id] = self._resolve(features, input_inventory.vector_features)
        return diagnosis_frame(rows)

    def _resolve(
        self, features: dict, vector_features: set
    ) -> tuple[str, str, tuple[str, ...]]:
        """
        Diagnose one asset from its features.

        Args:
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors.

        Returns:
            tuple[str, str, tuple[str, ...]]:
                The diagnosis, the detail, and the features that would
                each, alone, admit the asset.
        """
        year = features.get(self.year_built_key)
        if year is None:
            return (
                DIAGNOSIS_INVALID,
                f'no {self.year_built_key}',
                (self.year_built_key,),
            )
        if is_probabilistic_feature(self.year_built_key, year, vector_features):
            return (
                DIAGNOSIS_UNSUPPORTED,
                f'{self.year_built_key} holds one value per world; '
                'multi-world inference is planned and not supported yet',
                (),
            )
        failure = self._check_value(year)
        if failure is not None:
            return DIAGNOSIS_INVALID, failure, (self.year_built_key,)
        return DIAGNOSIS_OK, '', ()

    def _check_value(self, year: object) -> str | None:
        """
        Say why one year cannot be used.

        Args:
            year (object):
                The value read from the asset.

        Returns:
            str | None:
                The reason the value is unusable, or ``None`` where it is
                usable.
        """
        if isinstance(year, bool):
            return f'{self.year_built_key} is a boolean: {year!r}'
        if _outside_year_bounds(year):
            return (
                f'{self.year_built_key} of {year!r} sits outside the years '
                'the era comparison can represent'
            )
        try:
            value = float(year)
        except (TypeError, ValueError):
            return f'{self.year_built_key} is not a number: {year!r}'
        except OverflowError:
            return (
                f'{self.year_built_key} of {year!r} is outside the range '
                'the era comparison can read'
            )
        if not np.isfinite(value):
            return f'{self.year_built_key} is not a number: {year!r}'
        if value <= 0:
            return f'{self.year_built_key} is not a positive year: {year!r}'
        if value % 1 != 0:
            return f'{self.year_built_key} is not a whole year: {year!r}'
        return None

    def draw(self, input_inventory: AssetInventory) -> pd.Series:
        """
        Assign a design level to every asset, assuming all are usable.

        Args:
            input_inventory (AssetInventory):
                An inventory whose assets all passed :meth:`check`.

        Returns:
            pd.Series:
                Design levels indexed by asset id, named
                ``DesignLevel``.

        Raises:
            ValueError:
                If an asset carries no usable year, or the state or the
                seismic zone is not in the design level tables.
        """
        years = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            years[asset_id] = features.get(self.year_built_key)
        if not years:
            return pd.Series(dtype=object, name='DesignLevel')
        return assign_design_levels(
            pd.Series(years, dtype=object),
            self.state,
            self.seismic_zone,
            field_name=self.year_built_key,
        )

    def infer(self, input_inventory: AssetInventory) -> AssetInventory:
        """
        Assign a design level to every asset in the inventory.

        Args:
            input_inventory (AssetInventory):
                The inventory to process.

        Returns:
            AssetInventory:
                A copy of the input with the design level added under
                ``design_level_key``.

        Raises:
            ValueError:
                If any asset cannot be assigned a design level, naming
                the assets and their details.
        """
        verdicts = self.check(input_inventory)
        failed = verdicts.loc[verdicts['diagnosis'] != DIAGNOSIS_OK, 'detail']
        if not failed.empty:
            reasons = preview_failures(failed)
            raise ValueError(
                f'Cannot assign {self.design_level_key} for {len(failed)} '
                f'of {len(verdicts)} assets: {reasons}. Remove them or '
                'supply the missing data, then call infer() again.'
            )
        design_levels = self.draw(input_inventory)
        output_inventory = deepcopy(input_inventory)
        for asset_id, level in design_levels.items():
            output_inventory.add_asset_features(
                asset_id, {self.design_level_key: level}
            )
        return output_inventory
