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

# States with their own Appendix C matrix, keyed by the full state name
# that Table A-22 uses.
STATE_DESIGN_LEVELS = {'California': CA_DESIGN_LEVELS}


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


def _preview(asset_ids: list) -> str:
    """
    Format up to ten asset ids for an error message.

    Args:
        asset_ids (list):
            The asset ids to show.

    Returns:
        str:
            A comma-separated preview with a count of any remainder.
    """
    shown = ', '.join(str(asset_id) for asset_id in asset_ids[:10])
    extra = len(asset_ids) - 10
    if extra > 0:
        shown += f' and {extra} more'
    return shown


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
            non-numeric, non-integral, or not positive.
    """
    breaks, levels = _select_era_row(state, seismic_zone)
    years = pd.to_numeric(year_built, errors='coerce')
    invalid = years.isna() | (years <= 0) | (years % 1 != 0)
    if invalid.any():
        offenders = list(year_built.index[invalid])
        raise ValueError(
            f'{field_name} must be a positive whole year. Missing or '
            f'invalid values on {len(offenders)} of {len(year_built)} '
            f'assets: {_preview(offenders)}.'
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
                If an asset has no construction year, holds an invalid
                one, or carries it as a per-world list. Multi-world
                inference is planned and not supported yet.
        """
        years = {}
        missing = []
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            year = features.get(self.year_built_key)
            if year is None:
                missing.append(asset_id)
            elif is_probabilistic_feature(
                self.year_built_key, year, input_inventory.vector_features
            ):
                raise ValueError(
                    f'Asset {asset_id} carries {self.year_built_key} as '
                    'a per-world list. Multi-world inference is planned '
                    'and not supported yet.'
                )
            else:
                years[asset_id] = year
        if missing:
            raise ValueError(
                f'Every asset needs {self.year_built_key} for the design '
                f'level. Missing on: {_preview(missing)}.'
            )
        output_inventory = deepcopy(input_inventory)
        if years:
            design_levels = assign_design_levels(
                pd.Series(years),
                self.state,
                self.seismic_zone,
                field_name=self.year_built_key,
            )
            for asset_id, level in design_levels.items():
                output_inventory.add_asset_features(
                    asset_id, {self.design_level_key: level}
                )
        return output_inventory
