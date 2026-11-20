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
Read census block groups from the TIGER/Line files.

.. autosummary::

      TIGERLineScraper
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd

from brails.utils.tiger_files import DEFAULT_TIGER_VINTAGE, read_tiger_file

if TYPE_CHECKING:
    from brails.types.region_boundary import RegionBoundary

# The column of the block groups that holds the name of their state.
STATE_NAME_COLUMN = 'STATE_NAME'


class TIGERLineScraper:
    """
    Read the census block groups of a region from the TIGER/Line files of a release.

    The national state file and the block-group file of each state the region
    intersects are read with :func:`~brails.utils.tiger_files.read_tiger_file`, which
    keeps each download in memory for the process.

    Args:
        vintage (int, optional):
            The year of the TIGER/Line release. Defaults to
            :data:`~brails.utils.tiger_files.DEFAULT_TIGER_VINTAGE`.
        timeout (float, optional):
            Passed to :func:`~brails.utils.tiger_files.read_tiger_file`. Defaults to
            120.
        retries (int, optional):
            Passed to :func:`~brails.utils.tiger_files.read_tiger_file`. Defaults to
            3.
    """

    def __init__(
        self,
        vintage: int = DEFAULT_TIGER_VINTAGE,
        *,
        timeout: float = 120,
        retries: int = 3,
    ) -> None:
        """Keep the release and the download settings."""
        self.vintage = vintage
        self.timeout = timeout
        self.retries = retries

    def get_block_groups(self, region: RegionBoundary) -> gpd.GeoDataFrame:
        """
        Return the block groups whose area overlaps a region.

        A block group or a state that shares only an edge or a point with the region
        is left out, so a county or place of the same release does not bring in its
        neighbors. Before a state's block-group file is downloaded, a line names the
        state.

        Args:
            region (RegionBoundary):
                The region, whose boundary is in EPSG:4326.

        Returns:
            geopandas.GeoDataFrame:
                The rows of the block groups in the TIGER/Line files, in ascending
                order of ``GEOID`` and in the files' CRS, NAD83 (EPSG:4269), with the
                name of each block group's state in :data:`STATE_NAME_COLUMN`.

        Raises:
            ValueError:
                If the region overlaps no state of the release's state file, or if
                the server has no file of the release, naming its URL and vintage.
            requests.RequestException:
                If a download fails otherwise, as
                :func:`~brails.utils.tiger_files.read_tiger_file` raises it.
        """
        boundary, _, _ = region.get_boundary()
        download = {
            'vintage': self.vintage,
            'timeout': self.timeout,
            'retries': self.retries,
        }
        states = read_tiger_file('STATE', 'us', **download)
        area = gpd.GeoSeries([boundary], crs='EPSG:4326').to_crs(states.crs).iloc[0]
        states = states[states.intersects(area) & ~states.touches(area)]
        states = states.sort_values('STATEFP')
        if states.empty:
            msg = (
                f'The region lies in no state of the TIGER/Line {self.vintage} '
                'state file. Check that its boundary is in longitude and latitude '
                '(EPSG:4326) and lies in the United States or its territories.'
            )
            raise ValueError(msg)
        frames = []
        for state, name in zip(states['STATEFP'], states['NAME']):
            notice = f'Reading the TIGER/Line {self.vintage} block groups of {name}.'
            block_groups = read_tiger_file('BG', state, notice=notice, **download)
            overlap = block_groups.intersects(area) & ~block_groups.touches(area)
            inside = block_groups[overlap]
            frames.append(inside.assign(**{STATE_NAME_COLUMN: name}))
        return pd.concat(frames).sort_values('GEOID', ignore_index=True)
