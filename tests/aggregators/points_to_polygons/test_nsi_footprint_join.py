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

"""Tests for the join of NSI points to building footprints."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pandas as pd
import pytest
import shapely
from pyproj import CRS, Transformer

from brails.aggregators import NSIFootprintJoin
from brails.aggregators.points_to_polygons import nsi_footprint_join as nfj
from brails.scrapers import NSIFileReader
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils import Importer
from brails.utils.deterministic_sampling import asset_key_digest, keyed_uniforms
from brails.utils.unit_count_rules import point_class
from tests.imputers.donor_fixtures import features_of
from tests.scrapers.data.nsi_pooled_fixture_generator import (
    FIXTURE_PATH as NSI_SAMPLE,
)

# Test geometry is laid out in meters in UTM zone 10N near San Francisco and stored
# in EPSG:4326, as the join receives it.
METRIC_CRS = 'EPSG:32610'
X0, Y0 = 550_000.0, 4_180_000.0
_TO_LONLAT = Transformer.from_crs(METRIC_CRS, 'EPSG:4326', always_xy=True)
SEED = 20261027


def lonlat(x: float, y: float) -> list[float]:
    """Return the longitude and latitude of a local metric position."""
    lon, lat = _TO_LONLAT.transform(X0 + x, Y0 + y)
    return [lon, lat]


def square(x: float, y: float, side: float = 10.0) -> list[list[float]]:
    """Return a closed square ring with its lower left corner at (x, y)."""
    corners = [(x, y), (x + side, y), (x + side, y + side), (x, y + side), (x, y)]
    return [lonlat(cx, cy) for cx, cy in corners]


# Exact binary coordinates, so points on an edge lie on it exactly.
UNIT_DEG = 1 / 4096


def footprints(shapes: dict[Any, tuple[float, float, float]]) -> AssetInventory:
    """Build a footprint inventory from ``{id: (x, y, side)}``."""
    inventory = AssetInventory()
    for asset_id, (x, y, side) in shapes.items():
        inventory.add_asset(
            asset_id, Asset(asset_id, square(x, y, side), {'type': 'building'})
        )
    return inventory


def nsi(
    fd_id: int,
    x: float,
    y: float,
    occtype: str,
    resunits: int | None = None,
    **fields: Any,
) -> tuple[int, list[float], dict[str, Any]]:
    """Describe one NSI record at a local metric position."""
    features = {'fd_id': fd_id, 'occtype': occtype, 'cbfips': '060750101001'}
    if resunits is not None:
        features['resunits'] = resunits
    # The reader omits a null cell, so a field given as None is left out.
    features.update(
        {name: value for name, value in fields.items() if value is not None}
    )
    return fd_id, lonlat(x, y), features


def points(*records: tuple[int, list[float], dict[str, Any]]) -> AssetInventory:
    """Build an NSI point inventory keyed by fd_id."""
    inventory = AssetInventory()
    for fd_id, coordinate, given in records:
        features = dict(given)
        features.setdefault('x', coordinate[0])
        features.setdefault('y', coordinate[1])
        features.setdefault('type', 'building')
        inventory.add_asset(fd_id, Asset(fd_id, [list(coordinate)], features))
    return inventory


def run(
    shapes: AssetInventory,
    records: AssetInventory,
    seed: int = SEED,
    cap_m: float = 15.0,
) -> nfj.NSIJoinResult:
    """Run the join in the test CRS."""
    return NSIFootprintJoin(cap_m=cap_m, seed=seed, projected_crs=METRIC_CRS).join(
        shapes, records
    )


def link_of(result: nfj.NSIJoinResult) -> dict[Any, Any]:
    """Map each linked fd_id to its footprint."""
    return dict(zip(result.links['fd_id'], result.links['asset_id']))


def branch_of(result: nfj.NSIJoinResult) -> dict[Any, str]:
    """Map each linked fd_id to its branch."""
    return dict(zip(result.links['fd_id'], result.links['branch']))


def reason_of(result: nfj.NSIJoinResult) -> dict[Any, str]:
    """Map each unlinked fd_id to its reason."""
    return dict(zip(result.unlinked['fd_id'], result.unlinked['reason']))


def table(frame: pd.DataFrame) -> dict[str, list]:
    """Return a frame's columns as lists, with every missing value as None."""
    return {
        column: [None if pd.isna(value) else value for value in frame[column]]
        for column in frame.columns
    }


def canonical(result: nfj.NSIJoinResult) -> dict[str, Any]:
    """Return a comparable form of a join result."""
    return {
        'observed': [
            (key, asset.coordinates, sorted(asset.features.items()))
            for key, asset in result.observed.inventory.items()
        ],
        'derived': [
            (key, asset.coordinates, sorted(asset.features.items()))
            for key, asset in result.derived.inventory.items()
        ],
        'links': table(result.links),
        'unlinked': table(result.unlinked),
        'branch_counts': result.branch_counts.to_dict(),
    }


class TestClassification:
    """Parsing, the count rule and the walk groups."""

    @pytest.mark.parametrize(
        ('occtype', 'expected'),
        [
            ('RES1-1SNB', 'RES1'),
            ('RES3A', 'RES3A'),
            ('RES3', 'RES3A'),
            ('res3-x', 'RES3A'),
            ('COM1', 'COM1'),
            ('', None),
            (None, None),
        ],
    )
    def test_parse_occupancy_class(self, occtype: Any, expected: Any) -> None:
        """The class is the upper-case head, a bare RES3 read as RES3A."""
        assert nfj.parse_occupancy_class(occtype) == expected

    @pytest.mark.parametrize(
        ('occupancy', 'units', 'expected_class', 'expected_group'),
        [
            ('RES1', 1, 'RES1', 'single'),
            ('RES1', 4, 'RES3B', 'multi_family'),
            ('RES3A', 1, 'RES1', 'single'),
            ('RES3C', 7.0, 'RES3C', 'multi_family'),
            ('RES2', 3, 'RES2', 'single'),
            ('RES4', None, 'RES4', 'non_residential'),
            ('RES5', None, 'RES5', 'non_residential'),
            ('RES6', None, 'RES6', 'non_residential'),
            ('COM1', None, 'COM1', 'non_residential'),
        ],
    )
    def test_point_class_and_group(
        self, occupancy: str, units: Any, expected_class: str, expected_group: str
    ) -> None:
        """The count rule applies per record, and the group follows the class."""
        classified = point_class(occupancy, units)
        assert classified == expected_class
        assert nfj.walk_group(classified) == expected_group


class TestInsidePass:
    """Points that a footprint covers."""

    @pytest.mark.parametrize(('x', 'y'), [(10, 10), (10, 5)], ids=['vertex', 'edge'])
    def test_boundary_point_is_inside(self, x: float, y: float) -> None:
        """A point on a footprint's vertex or edge belongs to it."""
        shapes = AssetInventory()
        ring = [
            [-122.5 + dx * UNIT_DEG, 37.75 + dy * UNIT_DEG]
            for dx, dy in [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        ]
        shapes.add_asset('A', Asset('A', ring, {}))
        position = [-122.5 + x * UNIT_DEG, 37.75 + y * UNIT_DEG]
        result = run(shapes, points((1, position, {'occtype': 'RES1'})))
        assert branch_of(result) == {1: 'inside'}
        assert result.links['distance_m'].tolist() == [0.0]
        assert result.links['walk_position'].isna().all()

    def test_smallest_footprint_takes_the_point(self) -> None:
        """A point inside nested footprints goes to the smaller."""
        shapes = footprints({'A': (0, 0, 30), 'B': (5, 5, 10)})
        result = run(shapes, points(nsi(1, 8, 8, 'RES1')))
        assert link_of(result) == {1: 'B'}

    def test_nearest_centroid_breaks_an_area_tie(self) -> None:
        """Between equal areas, the footprint whose centroid is nearest wins."""
        # Two translated squares are equal in projected area only to rounding, so the
        # pass is called on hand-built arrays, with one set of coordinates for the
        # inside test and the measurements.
        polygons = np.asarray(
            [shapely.box(4, 0, 14, 10), shapely.box(0, 0, 10, 10)], dtype=object
        )
        point_geoms = np.asarray([shapely.Point(5, 5)], dtype=object)
        # The lowest rank is the farther footprint, so the rank cannot decide.
        rank = np.asarray([0, 1])
        inside = NSIFootprintJoin._inside_pass(
            polygons, polygons, point_geoms, point_geoms, rank
        )
        assert inside == {0: 1}

    def test_lowest_asset_id_breaks_a_full_tie(self) -> None:
        """Between identical footprints, the lowest asset id wins."""
        shapes = footprints({'fp_b': (0, 0, 10), 'fp_a': (0, 0, 10)})
        result = run(shapes, points(nsi(1, 5, 5, 'RES1')))
        assert link_of(result) == {1: 'fp_a'}

    def test_every_inside_point_is_assigned(self) -> None:
        """The inside pass assigns regardless of walk group."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(
            nsi(1, 2, 2, 'RES1'), nsi(2, 5, 5, 'COM1'), nsi(3, 8, 8, 'RES1')
        )
        result = run(shapes, records)
        assert link_of(result) == {1: 'A', 2: 'A', 3: 'A'}
        assert set(branch_of(result).values()) == {'inside'}


class TestAcceptance:
    """The walk's acceptance branches, candidate order and cap."""

    @pytest.mark.parametrize('occtype', ['RES1', 'RES3B', 'COM1', 'RES2'])
    def test_empty_footprint_accepts_any_point(self, occtype: str) -> None:
        """An empty footprint takes a point of any group."""
        shapes = footprints({'A': (0, 0, 10)})
        result = run(shapes, points(nsi(1, 15, 5, occtype, 4)))
        assert branch_of(result) == {1: 'walk_to_empty'}
        assert result.links['distance_m'].iloc[0] == pytest.approx(5.0)

    @pytest.mark.parametrize(
        ('held', 'arrival'),
        [
            ('RES3B', 'COM1'),
            ('COM1', 'RES3B'),
            ('RES3B', 'RES3C'),
            ('COM1', 'IND2'),
            ('RES3C', 'RES4'),
            ('RES3C', 'RES5'),
            ('RES3C', 'RES6'),
        ],
    )
    def test_shared_footprint_accepts_multi_family_and_non_residential(
        self, held: str, arrival: str
    ) -> None:
        """Multi-family and non-residential points join each other both ways."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(1, 5, 5, held, 4), nsi(2, 15, 5, arrival, 8))
        result = run(shapes, records)
        assert branch_of(result) == {1: 'inside', 2: 'walk_to_shared'}

    @pytest.mark.parametrize('arrival', ['RES1', 'RES2', 'RES3B', 'COM1'])
    @pytest.mark.parametrize('held', ['RES1', 'RES2'])
    def test_single_dwelling_accepts_nothing(self, held: str, arrival: str) -> None:
        """A footprint whose class is a single dwelling refuses every arrival."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(1, 5, 5, held, 1), nsi(2, 15, 5, arrival, 4))
        result = run(shapes, records)
        assert link_of(result) == {1: 'A'}
        assert reason_of(result) == {2: 'refused'}

    @pytest.mark.parametrize(
        ('arrival', 'resunits', 'branch'),
        [
            ('RES3C', 6, 'walk_to_shared'),
            ('COM1', None, 'walk_to_shared'),
            ('RES1', 1, None),
        ],
    )
    def test_footprint_class_decides_acceptance(
        self, arrival: str, resunits: int | None, branch: str | None
    ) -> None:
        """Two one-unit dwellings make a RES3A footprint, which is not single."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(
            nsi(1, 3, 3, 'RES1', 1),
            nsi(2, 6, 6, 'RES1', 1),
            nsi(3, 15, 5, arrival, resunits),
        )
        result = run(shapes, records)
        assert branch_of(result).get(3) == branch
        assert reason_of(result) == ({} if branch else {3: 'refused'})

    @pytest.mark.parametrize(
        ('res2_sqft', 'branch'), [(500.0, 'walk_to_shared'), (2000.0, None)]
    )
    def test_governing_res2_keeps_a_footprint_single(
        self, res2_sqft: float, branch: str | None
    ) -> None:
        """Beside a RES3B record, a RES2 record closes the footprint if it governs."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(
            nsi(1, 3, 3, 'RES2', 1, sqft=res2_sqft),
            nsi(2, 6, 6, 'RES3B', 4, sqft=900.0),
            nsi(3, 15, 5, 'RES3C', 6),
        )
        result = run(shapes, records)
        assert branch_of(result).get(3) == branch
        assert reason_of(result) == ({} if branch else {3: 'refused'})

    @pytest.mark.parametrize('held', ['RES3B', 'COM1'])
    @pytest.mark.parametrize('arrival', ['RES1', 'RES2'])
    def test_shared_footprint_refuses_a_single_dwelling(
        self, held: str, arrival: str
    ) -> None:
        """A single dwelling does not join a multi-family or non-residential one."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(1, 5, 5, held, 4), nsi(2, 15, 5, arrival, 1))
        result = run(shapes, records)
        assert reason_of(result) == {2: 'refused'}

    def test_one_unit_multi_family_record_walks_as_single(self) -> None:
        """A one-unit RES3A record is a single dwelling after the count rule."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(1, 5, 5, 'COM1'), nsi(2, 15, 5, 'RES3A', 1))
        result = run(shapes, records)
        assert reason_of(result) == {2: 'refused'}
        assert result.unlinked['point_class'].tolist() == ['RES1']

    def test_refused_point_takes_the_next_footprint(self) -> None:
        """A point turned away tries the next nearest footprint in the cap."""
        shapes = footprints({'A': (0, 0, 10), 'B': (24, 0, 10)})
        records = points(nsi(1, 5, 5, 'RES1'), nsi(2, 14, 5, 'RES1'))
        result = run(shapes, records)
        assert link_of(result) == {1: 'A', 2: 'B'}
        row = result.links.set_index('fd_id').loc[2]
        assert row['branch'] == 'walk_to_empty'
        assert row['distance_m'] == pytest.approx(10.0)

    def test_nearest_footprint_first(self) -> None:
        """Of two empty footprints, the nearer takes the point."""
        shapes = footprints({'far': (22, 0, 10), 'near': (0, 0, 10)})
        result = run(shapes, points(nsi(1, 14, 5, 'RES1')))
        assert link_of(result) == {1: 'near'}

    def test_distance_tie_goes_to_the_lowest_asset_id(self) -> None:
        """Equidistant footprints are taken in ascending asset id."""
        shapes = footprints({'fp_b': (20, 0, 10), 'fp_a': (0, 0, 10)})
        result = run(shapes, points(nsi(1, 15, 5, 'RES1')))
        assert link_of(result) == {1: 'fp_a'}

    def test_walk_arrival_joins_an_earlier_walk_arrival(self) -> None:
        """A footprint the walk filled accepts a second shared-group point."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(1, 14, 5, 'COM1'), nsi(2, -4, 5, 'RES3C', 6))
        result = run(shapes, records)
        assert link_of(result) == {1: 'A', 2: 'A'}
        assert sorted(branch_of(result).values()) == [
            'walk_to_empty',
            'walk_to_shared',
        ]

    def test_cap_is_inclusive(self) -> None:
        """A footprint exactly at the cap is a candidate; one beyond is not."""
        points_m = np.asarray([shapely.Point(25, 5)], dtype=object)
        polygons_m = np.asarray([shapely.box(0, 0, 10, 10)], dtype=object)
        rank = np.asarray([0])
        at_cap = NSIFootprintJoin(cap_m=15.0, seed=SEED)
        assert at_cap._candidates([0], points_m, polygons_m, rank) == {
            0: [(0, 15.0)]
        }
        below = NSIFootprintJoin(cap_m=14.999, seed=SEED)
        assert below._candidates([0], points_m, polygons_m, rank) == {}

    def test_cap(self) -> None:
        """A footprint beyond the cap is not reached; a wider cap reaches it."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(1, 26, 5, 'RES1'))
        assert reason_of(run(shapes, records)) == {1: 'no_footprint_within_cap'}
        assert link_of(run(shapes, records, cap_m=20.0)) == {1: 'A'}

    def test_walk_arrival_closes_a_footprint(self) -> None:
        """A single dwelling brought by the walk closes the footprint."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(1, 13, 5, 'RES1'), nsi(2, -3, 5, 'RES1'))
        result = run(shapes, records)
        assert len(result.links) == 1
        assert result.unlinked['reason'].tolist() == ['refused']

    def test_walk_arrival_into_an_empty_footprint_sets_its_class(self) -> None:
        """A single dwelling that walks into an empty footprint makes it single."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(2, 13, 5, 'RES1', 1), nsi(1, -3, 5, 'RES3B', 4))
        result = run(shapes, records)
        assert branch_of(result) == {2: 'walk_to_empty'}
        assert reason_of(result) == {1: 'refused'}


class TestUnlinked:
    """The unlinked rows and the accounting of every point."""

    def test_unlinked_rows_carry_their_units(self) -> None:
        """An unlinked point is a row with its class, units and place."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(
            nsi(1, 5, 5, 'RES1'),
            nsi(2, 15, 5, 'RES3B', 0),
            nsi(3, 100, 100, 'COM1', 0),
            nsi(4, 100, -100, 'RES3D', 12),
        )
        result = run(shapes, records)
        frame = result.unlinked.set_index('fd_id')
        assert list(result.unlinked.columns) == list(nfj.UNLINKED_COLUMNS)
        assert frame.loc[2, 'reason'] == 'refused'
        assert frame.loc[2, 'units'] == 3.5
        assert frame.loc[2, 'resunits'] == 0
        assert frame.loc[2, 'point_class'] == 'RES3B'
        assert frame.loc[2, 'occtype'] == 'RES3B'
        assert frame.loc[3, 'reason'] == 'no_footprint_within_cap'
        assert math.isnan(frame.loc[3, 'units'])
        assert frame.loc[4, 'units'] == 12.0
        assert frame.loc[4, 'cbfips'] == '060750101001'
        assert frame.loc[4, 'x'] == pytest.approx(lonlat(100, -100)[0])
        assert frame.loc[4, 'y'] == pytest.approx(lonlat(100, -100)[1])

    def test_every_point_is_accounted_for_once(self, mixed: tuple) -> None:
        """Links and unlinked rows partition the points."""
        shapes, records = mixed
        result = run(shapes, records)
        linked = set(result.links['fd_id'])
        unlinked = set(result.unlinked['fd_id'])
        assert not linked & unlinked
        assert linked | unlinked == set(records.inventory)
        assert result.links['fd_id'].is_unique


@pytest.fixture
def mixed() -> tuple[AssetInventory, AssetInventory]:
    """Twenty footprints and sixty points over the three walk groups, at random."""
    generator = random.Random(4)
    shapes = footprints(
        {
            f'fp{index:02d}': (
                generator.uniform(0, 120),
                generator.uniform(0, 120),
                10,
            )
            for index in range(20)
        }
    )
    classes = ['RES1', 'RES1', 'RES3A', 'RES3C', 'COM1', 'RES4', 'RES2', 'IND2']
    records = points(
        *(
            nsi(
                1000 + index,
                generator.uniform(-10, 140),
                generator.uniform(-10, 140),
                generator.choice(classes),
                generator.choice([None, 0, 1, 2, 6]),
                sqft=generator.choice([None, 800.0, 2400.0]),
                fullrep=generator.choice([None, 1.5e5, 4.0e5]),
            )
            for index in range(60)
        )
    )
    return shapes, records


def reordered(inventory: AssetInventory, seed: int) -> AssetInventory:
    """Return a copy of an inventory with its assets inserted in another order."""
    keys = list(inventory.inventory)
    random.Random(seed).shuffle(keys)
    copy = AssetInventory()
    for key in keys:
        copy.add_asset(key, inventory.inventory[key])
    copy.n_pw = inventory.n_pw
    return copy


class TestOrderedPass:
    """The properties an ordered assignment pass must hold."""

    @staticmethod
    def contest() -> tuple[AssetInventory, AssetInventory]:
        """Two single dwellings within the cap of one empty footprint."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(101, 13, 5, 'RES1'), nsi(102, -6, 5, 'RES1'))
        return shapes, records

    def test_golden_assignment(self) -> None:
        """Under seed 10 the farther point comes first and takes the footprint."""
        result = run(*self.contest(), seed=10)
        assert table(result.links) == {
            'fd_id': [102],
            'asset_id': ['A'],
            'branch': ['walk_to_empty'],
            'distance_m': [pytest.approx(6.0)],
            'walk_position': [0],
        }
        assert table(result.unlinked)['fd_id'] == [101]
        assert table(result.unlinked)['reason'] == ['refused']
        uniforms = NSIFootprintJoin(seed=10).walk_uniforms([101, 102])
        assert uniforms.tolist() == [0.6303956087766464, 0.16863599856001843]

    def test_seed_changes_the_order(self) -> None:
        """Different seeds give both outcomes of the contest."""
        winners = {
            run(*self.contest(), seed=seed).links['fd_id'].iloc[0]
            for seed in range(40)
        }
        assert winners == {101, 102}

    def test_order_key_survives_subsetting(self) -> None:
        """A point's walk uniform does not depend on the other points."""
        join = NSIFootprintJoin(seed=SEED)
        full = join.walk_uniforms([5, 9, 12, 40])
        for position, fd_id in enumerate([5, 9, 12, 40]):
            assert join.walk_uniforms([fd_id])[0] == full[position]

    def test_order_key_is_drawn_at_world_zero(self) -> None:
        """The walk uniform is the keyed uniform of world 0 under its label."""
        ids = [5, 9, 12, 40]
        join = NSIFootprintJoin(seed=SEED)
        expected = keyed_uniforms(
            asset_key_digest(ids),
            seed=SEED,
            purpose='nsi_footprint_walk',
            world=0,
        )
        np.testing.assert_array_equal(join.walk_uniforms(ids), expected)
        other_world = keyed_uniforms(
            asset_key_digest(ids), seed=SEED, purpose='nsi_footprint_walk', world=1
        )
        assert not np.array_equal(join.walk_uniforms(ids), other_world)

    @pytest.mark.parametrize('shuffle_seed', [1, 2, 3])
    def test_result_survives_reordering_of_points(
        self, mixed: tuple, shuffle_seed: int
    ) -> None:
        """Inserting the points in another order changes nothing."""
        shapes, records = mixed
        base = canonical(run(shapes, records))
        assert canonical(run(shapes, reordered(records, shuffle_seed))) == base

    @pytest.mark.parametrize('shuffle_seed', [1, 2, 3])
    def test_result_survives_reordering_of_footprints(
        self, mixed: tuple, shuffle_seed: int
    ) -> None:
        """Inserting the footprints in another order changes nothing."""
        shapes, records = mixed
        base = canonical(run(shapes, records))
        assert canonical(run(reordered(shapes, shuffle_seed), records)) == base

    def test_same_join_in_two_worlds(self, mixed: tuple) -> None:
        """Footprints declaring two worlds give the join of one world."""
        shapes, records = mixed
        two_worlds = AssetInventory()
        for key, asset in shapes.inventory.items():
            features = dict(asset.features, NumberOfStories=[1, 2])
            two_worlds.add_asset(key, Asset(key, asset.coordinates, features))
        two_worlds.n_pw = 2
        two_worlds.validate_possible_worlds()
        assert canonical(run(two_worlds, records)) == canonical(run(shapes, records))

    def test_drawn_seed_is_recorded_and_replays(self, mixed: tuple) -> None:
        """Without a seed the join draws one, records it, and can be replayed."""
        shapes, records = mixed
        join = NSIFootprintJoin(projected_crs=METRIC_CRS)
        assert isinstance(join.seed, int)
        first = join.join(shapes, records)
        assert first.seed == join.seed
        replay = NSIFootprintJoin(seed=first.seed, projected_crs=METRIC_CRS)
        assert canonical(replay.join(shapes, records)) == canonical(first)


class TestPerFootprint:
    """Units, class, governing record and the two containers."""

    def test_no_housing_record_leaves_units_unset(self) -> None:
        """A footprint without a housing record gets status none only."""
        shapes = footprints({'A': (0, 0, 10)})
        result = run(shapes, points(nsi(1, 5, 5, 'COM1', 0)))
        observed = features_of(result.observed)['A']
        assert observed['NumberOfUnitsStatus'] == 'none'
        assert 'NumberOfUnits' not in observed
        assert observed['OccupancyClass'] == 'COM1'
        assert 'A' not in result.derived.inventory

    def test_counted_units_and_count_band_class(self) -> None:
        """Counted units are observed; the band class is derived."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(1, 3, 3, 'RES3A', 2), nsi(2, 6, 6, 'RES3A', 3))
        result = run(shapes, records)
        observed = features_of(result.observed)['A']
        derived = features_of(result.derived)['A']
        assert observed['NumberOfUnits'] == 5
        assert observed['NumberOfUnitsStatus'] == 'counted'
        assert observed['NSIOccupancyClass'] == 'RES3A'
        assert 'OccupancyClass' not in observed
        assert derived == {'OccupancyClass': 'RES3C'}

    def test_midpoint_filled_units_are_derived(self) -> None:
        """A unit count holding a midpoint goes to the derived container."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(1, 3, 3, 'RES3B', 0), nsi(2, 6, 6, 'RES3A', 2))
        result = run(shapes, records)
        observed = features_of(result.observed)['A']
        derived = features_of(result.derived)['A']
        assert observed['NumberOfUnitsStatus'] == 'midpoint_filled'
        assert 'NumberOfUnits' not in observed
        assert derived['NumberOfUnits'] == 5.5
        assert derived['OccupancyClass'] == 'RES3C'

    def test_housing_governs_before_gross_area(self) -> None:
        """A housing record governs a larger non-residential one."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(
            nsi(1, 3, 3, 'COM1', 0, sqft=9000.0),
            nsi(2, 6, 6, 'RES3B', 4, sqft=900.0),
        )
        observed = features_of(run(shapes, records).observed)['A']
        assert observed['NSIGoverningRecord'] == 2
        assert observed['NSIOccupancyClass'] == 'RES3B'

    def test_largest_gross_area_then_lowest_fd_id(self) -> None:
        """Among housing records, the largest sqft governs, then the lowest id."""
        shapes = footprints({'A': (0, 0, 10), 'B': (40, 0, 10)})
        records = points(
            nsi(1, 3, 3, 'RES3A', 2, sqft=800.0),
            nsi(2, 6, 6, 'RES3B', 4, sqft=1200.0),
            nsi(7, 43, 3, 'RES3A', 2, sqft=500.0),
            nsi(5, 46, 6, 'RES3A', 2, sqft=500.0),
        )
        observed = features_of(run(shapes, records).observed)
        assert observed['A']['NSIGoverningRecord'] == 2
        assert observed['B']['NSIGoverningRecord'] == 5

    def test_walk_arrival_can_govern_an_inside_record(self) -> None:
        """Inside and walk records are ranked by one key: housing first."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(
            nsi(1, 5, 5, 'COM1', 0, sqft=500.0),
            nsi(2, 14, 5, 'RES3C', 6, sqft=9000.0),
        )
        result = run(shapes, records)
        observed = features_of(result.observed)['A']
        assert branch_of(result) == {1: 'inside', 2: 'walk_to_shared'}
        assert observed['NSIGoverningRecord'] == 2
        assert observed['NSIOccupancyClass'] == 'RES3C'
        assert observed['NumberOfUnits'] == 6
        assert features_of(result.derived)['A']['OccupancyClass'] == 'RES3C'
        assert observed['NSIReach'] == 'inside'
        assert observed['NSIRecords'] == '1;2'

    def test_housing_governs_a_non_housing_first_arrival(self) -> None:
        """The order of arrival does not choose the governing record."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(
            nsi(11, 14, 5, 'COM1', sqft=9000.0),
            nsi(12, -4, 5, 'RES3B', 4, sqft=900.0),
        )
        result = run(shapes, records)
        positions = dict(zip(result.links['fd_id'], result.links['walk_position']))
        # Under SEED the non-housing record walks first.
        assert positions == {11: 0, 12: 1}
        observed = features_of(result.observed)['A']
        assert observed['NSIGoverningRecord'] == 12
        assert observed['NSIReach'] == 'walk'
        assert features_of(result.derived)['A']['OccupancyClass'] == 'RES3B'

    def test_unstated_gross_area_ranks_last(self) -> None:
        """A record whose sqft is not a number ranks below a stated one."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(
            nsi(1, 3, 3, 'RES3A', 2, sqft='1800'),
            nsi(2, 6, 6, 'RES3A', 2, sqft=500.0),
        )
        result = run(shapes, records)
        observed = features_of(result.observed)['A']
        assert observed['NSIGoverningRecord'] == 2
        assert observed['GrossArea'] == 500.0

    def test_res2_keeps_its_class(self) -> None:
        """A RES2 governing record keeps RES2 whatever its units."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(1, 3, 3, 'RES2', 1), nsi(2, 6, 6, 'RES2', 2))
        result = run(shapes, records)
        observed = features_of(result.observed)['A']
        assert observed['OccupancyClass'] == 'RES2'
        assert observed['NumberOfUnits'] == 3
        assert 'A' not in result.derived.inventory

    def test_sums_skip_null_records(self) -> None:
        """Area and value are summed over the records that state them."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(
            nsi(1, 2, 2, 'COM1', sqft=1000.0, fullrep=1.0e5),
            nsi(2, 5, 5, 'COM2', fullrep=2.0e5),
            nsi(3, 8, 8, 'COM3', sqft=500.0),
        )
        result = run(shapes, records)
        observed = features_of(result.observed)['A']
        derived = features_of(result.derived)['A']
        assert derived['GrossArea'] == 1500.0
        assert derived['ReplacementCost'] == 3.0e5
        assert 'GrossArea' not in observed
        assert 'ReplacementCost' not in observed

    def test_one_stated_value_is_observed(self) -> None:
        """A value one record states is NSI's own, in the observed container."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(
            nsi(1, 2, 2, 'COM1', fullrep=2.5e5, val_struct=1.0e5),
            nsi(2, 5, 5, 'COM2', sqft=700.0),
        )
        result = run(shapes, records)
        observed = features_of(result.observed)['A']
        assert observed['GrossArea'] == 700.0
        assert observed['ReplacementCost'] == 2.5e5
        assert 'A' not in result.derived.inventory

    def test_no_stated_value_writes_none(self) -> None:
        """A footprint whose records state no area gets no area."""
        shapes = footprints({'A': (0, 0, 10)})
        result = run(shapes, points(nsi(1, 2, 2, 'COM1')))
        assert 'GrossArea' not in features_of(result.observed)['A']
        assert 'ReplacementCost' not in features_of(result.observed)['A']

    def test_observed_carries_the_governing_record(self) -> None:
        """NSI's fields arrive under SimCenter names, without the position."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(
            nsi(1, 5, 5, 'RES1-1SWB', 1, med_yr_blt=1925, num_story=2, sqft=1600.0)
        )
        result = run(shapes, records)
        observed = features_of(result.observed)['A']
        assert observed['YearBuiltTractMedian'] == 1925
        assert observed['NumberOfStories'] == 2
        assert observed['Basement'] is True
        assert observed['NSIOccupancyClass'] == 'RES1'
        assert observed['GrossArea'] == 1600.0
        assert observed['cbfips'] == '060750101001'
        for name in ('Longitude', 'Latitude', 'resunits', 'fd_id'):
            assert name not in observed
        assert (
            result.observed.inventory['A'].coordinates
            == shapes.inventory['A'].coordinates
        )

    def test_inventories_follow_ascending_asset_id(self, mixed: tuple) -> None:
        """Both containers insert assets in ascending asset id."""
        shapes, records = mixed
        result = run(reordered(shapes, 9), records)
        for inventory in (result.observed, result.derived):
            keys = list(inventory.inventory)
            assert keys == sorted(keys)
        assert set(result.derived.inventory) <= set(result.observed.inventory)
        assert set(result.observed.inventory) == set(result.links['asset_id'])


class TestBranchCounts:
    """The branch table."""

    def test_counts_by_branch_and_group(self) -> None:
        """Each point is counted once under its branch and group."""
        shapes = footprints({'A': (0, 0, 10), 'B': (40, 0, 10), 'C': (80, 0, 10)})
        records = points(
            nsi(1, 5, 5, 'RES1'),
            nsi(2, 14, 5, 'RES1'),
            nsi(3, 45, 5, 'COM1'),
            nsi(4, 54, 5, 'RES3C', 6),
            nsi(5, 94, 5, 'RES3B', 3),
            nsi(6, 300, 300, 'COM1'),
            nsi(7, 300, 300, 'RES3A', 2),
        )
        counts = run(shapes, records).branch_counts
        assert list(counts.index) == list(nfj.BRANCH_COUNT_ROWS)
        assert list(counts.columns) == [
            'single',
            'multi_family',
            'non_residential',
            'all',
        ]
        assert counts.loc['inside'].to_dict() == {
            'single': 1,
            'multi_family': 0,
            'non_residential': 1,
            'all': 2,
        }
        assert counts.loc['walk_to_empty', 'multi_family'] == 1
        assert counts.loc['walk_to_shared', 'multi_family'] == 1
        assert counts.loc['unlinked_refused', 'single'] == 1
        assert counts.loc['unlinked_no_footprint_within_cap', 'all'] == 2
        assert counts.loc['stacked_outside_footprints'].to_dict() == {
            'single': 0,
            'multi_family': 1,
            'non_residential': 1,
            'all': 2,
        }
        assert counts.iloc[:-1]['all'].sum() == len(records.inventory)

    def test_stack_inside_a_footprint_is_not_counted(self) -> None:
        """Stacked records inside a footprint are not in the stacked row."""
        shapes = footprints({'A': (0, 0, 10)})
        records = points(nsi(1, 5, 5, 'COM1'), nsi(2, 5, 5, 'RES3B', 4))
        counts = run(shapes, records).branch_counts
        assert counts.loc['stacked_outside_footprints', 'all'] == 0
        assert counts.loc['inside', 'all'] == 2


class TestInputs:
    """Refused inputs and the configuration."""

    def test_rejects_non_inventories(self) -> None:
        """Both inputs must be asset inventories."""
        shapes = footprints({'A': (0, 0, 10)})
        with pytest.raises(TypeError, match='not dict'):
            run({}, points(nsi(1, 5, 5, 'RES1')))
        with pytest.raises(TypeError, match='not list'):
            run(shapes, [])

    def test_rejects_empty_inventories(self) -> None:
        """An empty input raises."""
        with pytest.raises(ValueError, match='footprint inventory is empty'):
            run(AssetInventory(), points(nsi(1, 5, 5, 'RES1')))
        with pytest.raises(ValueError, match='point inventory is empty'):
            run(footprints({'A': (0, 0, 10)}), AssetInventory())

    def test_rejects_a_footprint_that_is_not_a_polygon(self) -> None:
        """A point asset among the footprints raises, naming it."""
        shapes = footprints({'A': (0, 0, 10)})
        shapes.add_asset('P', Asset('P', [lonlat(50, 50)], {}))
        with pytest.raises(ValueError, match='not polygons: P'):
            run(shapes, points(nsi(1, 5, 5, 'RES1')))

    def test_rejects_a_point_that_is_not_a_point(self) -> None:
        """A polygon among the points raises, naming it."""
        records = points(nsi(1, 5, 5, 'RES1'))
        records.add_asset(9, Asset(9, square(0, 0), {'occtype': 'RES1'}))
        with pytest.raises(ValueError, match='not single points: 9'):
            run(footprints({'A': (0, 0, 10)}), records)

    def test_rejects_a_point_without_occupancy(self) -> None:
        """A point without an occtype raises, naming it."""
        records = points(nsi(1, 5, 5, 'RES1'))
        records.add_asset(2, Asset(2, [lonlat(6, 6)], {'fd_id': 2}))
        with pytest.raises(ValueError, match='no parseable occtype: 2'):
            run(footprints({'A': (0, 0, 10)}), records)

    def test_rejects_points_per_world(self) -> None:
        """NSI points held per world are refused."""
        records = points(nsi(1, 5, 5, 'RES1'))
        records.n_pw = 2
        with pytest.raises(ValueError, match='2 possible worlds'):
            run(footprints({'A': (0, 0, 10)}), records)

    @pytest.mark.parametrize('cap_m', [-1.0, math.inf, math.nan, '15'])
    def test_rejects_a_bad_cap(self, cap_m: Any) -> None:
        """The cap is a finite, non-negative number of meters."""
        with pytest.raises(ValueError, match='cap_m'):
            NSIFootprintJoin(cap_m=cap_m)

    def test_rejects_a_bad_seed(self) -> None:
        """A seed is an integer."""
        with pytest.raises(TypeError, match='seed'):
            NSIFootprintJoin(seed=True)
        with pytest.raises(TypeError, match='seed'):
            NSIFootprintJoin(seed=1.5)

    @pytest.mark.parametrize(
        ('crs', 'message'),
        [
            ('EPSG:4326', 'WGS 84'),
            ('EPSG:2227', 'US survey foot'),
            ('EPSG:3857', 'Pseudo-Mercator'),
            ('not a crs', 'not a CRS'),
        ],
    )
    def test_rejects_a_crs_not_in_meters(self, crs: str, message: str) -> None:
        """The distance CRS must be projected, in meters, and not Web Mercator."""
        with pytest.raises(ValueError, match=message):
            NSIFootprintJoin(projected_crs=crs)

    def test_estimates_a_utm_zone(self) -> None:
        """Without a CRS, the UTM zone of the footprints is used and returned."""
        shapes = footprints({'A': (0, 0, 10)})
        result = NSIFootprintJoin(seed=1).join(shapes, points(nsi(1, 15, 5, 'RES1')))
        assert result.crs == CRS.from_epsg(32610)
        assert result.links['distance_m'].iloc[0] == pytest.approx(5.0, abs=0.01)

    def test_discovered_by_the_importer(self) -> None:
        """The Importer finds the join class."""
        assert Importer().get_class('NSIFootprintJoin') is NSIFootprintJoin


class TestOnReaderOutput:
    """The join on NSI records as the parquet reader returns them."""

    def test_bay_area_sample(self) -> None:
        """Every record of the Bay Area sample is linked or unlinked, once."""
        records = NSIFileReader().read(NSI_SAMPLE)
        transformer = Transformer.from_crs('EPSG:4326', METRIC_CRS, always_xy=True)
        back = Transformer.from_crs(METRIC_CRS, 'EPSG:4326', always_xy=True)
        shapes = AssetInventory()
        for position, fd_id in enumerate(sorted(records.inventory)):
            if position % 3:
                continue
            x, y = transformer.transform(*records.inventory[fd_id].coordinates[0])
            ring = [
                list(back.transform(x + dx, y + dy))
                for dx, dy in ((-6, -6), (6, -6), (6, 6), (-6, 6), (-6, -6))
            ]
            shapes.add_asset(f'fp{fd_id}', Asset(f'fp{fd_id}', ring, {}))
        result = NSIFootprintJoin(seed=SEED, projected_crs=METRIC_CRS).join(
            shapes, records
        )
        linked = set(result.links['fd_id'])
        assert linked.isdisjoint(result.unlinked['fd_id'])
        assert linked | set(result.unlinked['fd_id']) == set(records.inventory)
        assert set(result.observed.inventory) == set(result.links['asset_id'])
        # Every footprint is centered on a record, which is inside it.
        branch = branch_of(result)
        for asset_id in shapes.inventory:
            assert branch[int(asset_id[2:])] == 'inside'
        walked = result.links[result.links['branch'] != 'inside']
        assert (walked['distance_m'] > 0).all()
        assert (walked['distance_m'] <= 15.0).all()
        counts = result.branch_counts
        assert counts.iloc[:-1]['all'].sum() == len(records.inventory)
