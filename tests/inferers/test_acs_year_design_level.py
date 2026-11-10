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

"""Tests for design levels assigned from years the ACS sampler draws."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from brails.inferers.acs_year_built_sampler.acs_year_built_sampler import (
    B25034_DECADES,
    B25127_CELLS,
    ACSYearBuiltSampler,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_design_level_inferer import (
    CA_DESIGN_LEVELS,
    HazusEQDesignLevelInferer,
    assign_design_levels,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from brails.types.asset_inventory import AssetInventory

SEED = 20261102
BLOCK_GROUP = '060010001001'
TRACT = '06001000100'

# B25034 units only in the 1940s and the 1970s, the decades holding the
# two era breaks of the zone 4 row of CA_DESIGN_LEVELS.
UNITS = {'1940 to 1949': 50.0, '1970 to 1979': 50.0}
BREAKS = CA_DESIGN_LEVELS['4'][0]


def test_drawn_years_give_the_zone_4_levels_per_world(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """
    Each world's drawn years get the zone 4 level on both sides of a break.

    The inferer runs at its defaults on each world's scalar inventory, and
    its levels equal those of ``assign_design_levels`` for California zone
    4 on the drawn years.
    """
    inventory = build_inventory(
        {
            f'r{k}': {
                'OccupancyClass': 'RES1',
                'NumberOfUnits': 1,
                'BLOCKGROUP_GEOID': BLOCK_GROUP,
            }
            for k in range(200)
        }
    )
    # block group that draws only in the two decades
    row = {decade[3]: UNITS.get(decade[0], 0.0) for decade in B25034_DECADES}
    b25034_bg = pd.DataFrame([row], index=pd.Index([BLOCK_GROUP], dtype=object))
    b25034_tract = pd.DataFrame([row], index=pd.Index([TRACT], dtype=object))
    cells = {v: 10.0 for _, _, pair in B25127_CELLS for v in pair}
    b25127 = pd.DataFrame([cells], index=pd.Index([TRACT], dtype=object))
    sides = set()
    for world in range(5):
        sampler = ACSYearBuiltSampler(
            b25034_bg, b25034_tract, b25127, seed=SEED, world=world
        )
        drawn = sampler.fit(inventory).infer(inventory)
        levels = HazusEQDesignLevelInferer().infer(drawn)
        ids = drawn.get_asset_ids()
        years = pd.Series(
            {k: drawn.get_asset_features(k)[1]['YearBuilt'] for k in ids}
        )
        expected = assign_design_levels(years, 'California', '4')
        for asset_id in ids:
            level = levels.get_asset_features(asset_id)[1]['DesignLevel']
            assert level == expected.loc[asset_id]
        for cut in BREAKS:
            near = years[(years // 10) == (cut // 10)]
            sides.update((cut, bool(year >= cut)) for year in near)
    assert sides == {(cut, side) for cut in BREAKS for side in (False, True)}

