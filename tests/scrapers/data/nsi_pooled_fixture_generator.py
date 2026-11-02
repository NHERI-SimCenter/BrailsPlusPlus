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
Write the stored sample NSI table.

The sample holds NSI 2026 records sampled from several San Francisco Bay
Area counties: the fields the NSI API serves and ``query_fips``, the county
request that served each row. The ``SAMPLE_SIZE`` records are shared about
evenly among the counties and chosen in ascending ``fd_id`` within each:

1. every record of the first ``STACKS_PER_COUNTY`` coordinates holding two to
   ``MAX_STACK_SIZE`` records each, so stacked mixed-use records are present;
2. the first ``H_PER_COUNTY`` records whose ``source`` is ``H`` not taken above;
3. the first records whose ``source`` is ``P`` not taken above, up to the
   county's share.

The selection depends only on the records of the table, not on its row
order. The module can be run by hand with the path of the NSI table to
sample:

    python tests/scrapers/data/nsi_pooled_fixture_generator.py NSI.parquet

The test suite imports only ``FIXTURE_PATH`` from this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

FIXTURE_PATH = Path(__file__).parent / 'nsi2026_bayarea_sample.parquet'
SAMPLE_SIZE = 200
STACKS_PER_COUNTY = 2
MAX_STACK_SIZE = 4
H_PER_COUNTY = 5


def cut_sample(pooled_path: str | Path) -> list[int]:
    """
    Choose the sampled records of an NSI table.

    Args:
        pooled_path (str or Path):
            The NSI parquet table to sample.

    Returns:
        list[int]:
            The ``fd_id`` of every sampled record, by county in FIPS order
            and in selection order within a county.
    """
    frame = pq.read_table(
        pooled_path, columns=['fd_id', 'x', 'y', 'source', 'query_fips']
    ).to_pandas()
    frame = frame.sort_values('fd_id', kind='stable')

    counties = sorted(frame['query_fips'].unique())
    base, extra = divmod(SAMPLE_SIZE, len(counties))
    chosen: list[int] = []
    for rank, county in enumerate(counties):
        share = base + (1 if rank < extra else 0)
        records = frame[frame['query_fips'] == county]
        taken: list[int] = []

        sizes = records.groupby(['x', 'y'])['fd_id'].transform('size')
        stacked = records[(sizes >= 2) & (sizes <= MAX_STACK_SIZE)]
        first_ids = stacked.groupby(['x', 'y'])['fd_id'].transform('min')
        for stack_id in sorted(first_ids.unique())[:STACKS_PER_COUNTY]:
            taken.extend(stacked.loc[first_ids == stack_id, 'fd_id'].tolist())

        rest = records[~records['fd_id'].isin(taken)]
        taken.extend(rest.loc[rest['source'] == 'H', 'fd_id'].head(H_PER_COUNTY))
        rest = records[~records['fd_id'].isin(taken)]
        taken.extend(
            rest.loc[rest['source'] == 'P', 'fd_id'].head(share - len(taken))
        )
        chosen.extend(int(fd_id) for fd_id in taken)

    if len(chosen) != SAMPLE_SIZE:
        message = f'The sample has {len(chosen)} records, not {SAMPLE_SIZE}.'
        raise RuntimeError(message)
    return chosen


def main() -> None:
    """Cut the sample from an NSI table and rewrite the stored fixture."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('pooled_path', help='the NSI parquet table to sample')
    args = parser.parse_args()

    chosen = cut_sample(args.pooled_path)
    table = pq.read_table(args.pooled_path)
    position = {fd_id: row for row, fd_id in enumerate(table['fd_id'].to_pylist())}
    sample = table.take([position[fd_id] for fd_id in chosen])
    pq.write_table(sample.replace_schema_metadata(None), FIXTURE_PATH)
    print(f'Wrote {sample.num_rows} records to {FIXTURE_PATH}.')


if __name__ == '__main__':
    main()
