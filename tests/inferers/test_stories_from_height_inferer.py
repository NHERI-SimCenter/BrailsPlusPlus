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

"""Tests for the inference of story counts from footprint heights."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import brails.inferers
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INFEASIBLE,
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
)
from brails.inferers.stories_from_height_inferer.stories_from_height_inferer import (
    HEIGHT_SCALES_FILE,
    STORIES_FROM_HEIGHT_KEY,
    StoriesFromHeightInferer,
)
from brails.utils import Importer

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from brails.types.asset_inventory import AssetInventory

MICROSOFT = 'Microsoft ML Buildings'
OSM = 'OpenStreetMap'

# Scales used by the tests, not the packaged calibration.
FIXTURE_SCALES = {
    'length_unit': 'm',
    'providers': {
        'overture': {
            'sources': {
                MICROSOFT: {'scale': 3.92, 'max_stories': 2},
                OSM: {'scale': 4.0, 'max_stories': None},
            }
        }
    },
}
STORIES = StoriesFromHeightInferer(length_unit='m', scales=FIXTURE_SCALES)

# The provenance of an Overture footprint whose height is OpenStreetMap's,
# and of one whose height is Microsoft's.
OSM_HEIGHT = {'footprint_provider': 'overture', 'sources': f'dataset: {OSM}'}
MICROSOFT_HEIGHT = {
    'footprint_provider': 'overture',
    'sources': f'dataset: {MICROSOFT}',
}

# Overture sources as a list of entries: the footprint is Microsoft's and
# its height OpenStreetMap's.
HEIGHT_FROM_OSM = [
    {'property': '', 'dataset': MICROSOFT},
    {'property': '/properties/height', 'dataset': OSM},
]


class TestRule:
    """A height converts to a story count by its source's scale."""

    def test_rounding(self) -> None:
        """1.5 scales round to two stories, 1.49 to one, and a low height to one."""
        assert STORIES.stories_of({'height': 6.0, **OSM_HEIGHT}) == 2
        assert STORIES.stories_of({'height': 5.96, **OSM_HEIGHT}) == 1
        assert STORIES.stories_of({'height': 0.5, **OSM_HEIGHT}) == 1

    def test_units(self) -> None:
        """Feet and meters give the same count."""
        in_feet = StoriesFromHeightInferer(length_unit='ft', scales=FIXTURE_SCALES)
        assert STORIES.stories_of({'height': 9.0, **OSM_HEIGHT}) == 2
        assert in_feet.stories_of({'height': 9.0 / 0.3048, **OSM_HEIGHT}) == 2

    def test_trusted_maximum(self) -> None:
        """At 3.92 m, a Microsoft height of 9.7 m gives two stories and 9.9 m none."""
        assert STORIES.stories_of({'height': 9.7, **MICROSOFT_HEIGHT}) == 2
        assert STORIES.stories_of({'height': 9.9, **MICROSOFT_HEIGHT}) is None

    @pytest.mark.parametrize(
        'height', [0.0, -3.0, 'NA', '9', True, None, [8.0, 9.0]]
    )
    def test_unusable_height(self, height: object) -> None:
        """A non-positive, missing, non-numeric or per-world height gives none."""
        assert STORIES.stories_of({'height': height, **OSM_HEIGHT}) is None


class TestProviderAndSource:
    """The provider keys the scales, and Overture's sources name the source."""

    def test_height_entry(self) -> None:
        """The entry whose property is the height names the source."""
        sources = (
            f'dataset: {OSM}, record_id: w1; '
            f'property: /properties/height, dataset: {MICROSOFT}, between: [0.0, 1.0]'
        )
        features = {
            'height': 9.9,
            'sources': sources,
            'footprint_provider': 'overture',
        }
        assert STORIES.stories_of(features) is None
        assert STORIES.stories_of({**features, 'height': 9.7}) == 2

    def test_first_whole_feature_entry(self) -> None:
        """Without a height entry, the first entry without a property names it."""
        features = {
            'height': 9.9,
            'footprint_provider': 'overture',
            'sources': f'dataset: {OSM}; dataset: {MICROSOFT}',
        }
        assert STORIES.stories_of(features) == 2
        reversed_sources = f'dataset: {MICROSOFT}; dataset: {OSM}'
        assert STORIES.stories_of({**features, 'sources': reversed_sources}) is None

    @pytest.mark.parametrize(
        'sources',
        [HEIGHT_FROM_OSM, np.array(HEIGHT_FROM_OSM, dtype=object)],
        ids=['list', 'array'],
    )
    def test_list_of_dictionaries(self, sources: object) -> None:
        """A list or array of dictionaries is read like the string."""
        features = {
            'height': 9.9,
            'sources': sources,
            'footprint_provider': 'overture',
        }
        assert STORIES.stories_of(features) == 2

    @pytest.mark.parametrize(
        ('provider', 'detail'),
        [
            ({}, 'no footprint_provider'),
            ({'footprint_provider': None}, 'no footprint_provider'),
            ({'footprint_provider': 'NA'}, 'no footprint_provider'),
            ({'footprint_provider': 5}, 'footprint_provider is not a string'),
        ],
    )
    def test_missing_provider(
        self,
        build_inventory: Callable[..., AssetInventory],
        provider: dict[str, Any],
        detail: str,
    ) -> None:
        """A sources string without a usable provider is invalid."""
        inventory = build_inventory(
            {'a': {'height': 9.0, 'sources': f'dataset: {OSM}', **provider}}
        )
        verdict = STORIES.check(inventory).loc['a']
        assert (verdict['diagnosis'], verdict['detail']) == (
            DIAGNOSIS_INVALID,
            detail,
        )

    def test_unlisted_provider(self) -> None:
        """A provider the scales do not list gives no count, whatever its sources."""
        features = {
            'height': 9.0,
            'footprint_provider': 'osm',
            'sources': f'dataset: {OSM}',
        }
        assert STORIES.stories_of(features) is None

    @pytest.mark.parametrize(
        'sources', ['NA', None, 'dataset: ', 'property: /properties/height']
    )
    def test_no_height_source(
        self, build_inventory: Callable[..., AssetInventory], sources: object
    ) -> None:
        """Overture sources that name no height source are invalid."""
        inventory = build_inventory(
            {
                'a': {
                    'height': 9.0,
                    'footprint_provider': 'overture',
                    'sources': sources,
                }
            }
        )
        verdict = STORIES.check(inventory).loc['a']
        assert (verdict['diagnosis'], verdict['detail']) == (
            DIAGNOSIS_INVALID,
            'no height source in sources',
        )

    def test_direct_form(self) -> None:
        """A provider's own scale applies to all its heights, sources unread."""
        scales = {
            'length_unit': 'm',
            'providers': {
                'osm': {'scale': 3.0, 'max_stories': None},
                'overture': {'scale': 3.0, 'max_stories': 2},
            },
        }
        direct = StoriesFromHeightInferer(length_unit='m', scales=scales)
        assert direct.stories_of({'height': 9.0, 'footprint_provider': 'osm'}) == 3
        assert direct.stories_of({'height': 6.0, **MICROSOFT_HEIGHT}) == 2
        assert (
            direct.stories_of({'height': 6.0, 'footprint_provider': 'overture'}) == 2
        )
        assert (
            direct.stories_of({'height': 9.0, 'footprint_provider': 'overture'})
            is None
        )


class TestScales:
    """Scales load from the packaged file, a path or a mapping and are checked."""

    def test_packaged_file_holds_only_what_the_calculation_reads(self) -> None:
        """The packaged file holds the unit and, per Overture source, two numbers."""
        with HEIGHT_SCALES_FILE.open(encoding='utf-8') as file:
            packaged = json.load(file)
        assert set(packaged) == {'length_unit', 'providers'}
        assert packaged['length_unit'] == 'm'
        assert set(packaged['providers']) == {'overture'}
        assert set(packaged['providers']['overture']) == {'sources'}
        for entry in packaged['providers']['overture']['sources'].values():
            assert set(entry) == {'scale', 'max_stories'}

    def test_packaged_sources_are_the_overture_datasets(self) -> None:
        """The packaged scales calibrate Overture's three height sources alone."""
        with HEIGHT_SCALES_FILE.open(encoding='utf-8') as file:
            sources = json.load(file)['providers']['overture']['sources']
        assert set(sources) == {MICROSOFT, OSM, 'USGS Lidar'}

    def test_packaged_default(self) -> None:
        """Without scales, the packaged file applies."""
        with HEIGHT_SCALES_FILE.open(encoding='utf-8') as file:
            scale = json.load(file)['providers']['overture']['sources'][OSM]['scale']
        default = StoriesFromHeightInferer(length_unit='m')
        assert default.stories_of({'height': 3 * scale, **OSM_HEIGHT}) == 3

    def test_path_and_mapping_forms(self, tmp_path: Path) -> None:
        """A file path and a mapping load alike."""
        path = tmp_path / 'scales.json'
        path.write_text(json.dumps(FIXTURE_SCALES), encoding='utf-8')
        counts = {
            StoriesFromHeightInferer(length_unit='m', scales=scales).stories_of(
                {'height': 13.0, **OSM_HEIGHT}
            )
            for scales in (path, str(path), FIXTURE_SCALES)
        }
        assert counts == {3}

    def test_the_scales_unit_is_read(self) -> None:
        """Scales in feet give the counts of the same scales in meters."""
        sources = FIXTURE_SCALES['providers']['overture']['sources']
        in_feet = StoriesFromHeightInferer(
            length_unit='m',
            scales={
                'length_unit': 'ft',
                'providers': {
                    'overture': {
                        'sources': {
                            source: {
                                'scale': entry['scale'] / 0.3048,
                                'max_stories': None,
                            }
                            for source, entry in sources.items()
                        }
                    }
                },
            },
        )
        for height in (4.0, 9.0, 13.0, 41.0):
            features = {'height': height, **OSM_HEIGHT}
            assert in_feet.stories_of(features) == STORIES.stories_of(features)

    @pytest.mark.parametrize(
        ('providers', 'match'),
        [
            ({'osm': {'scale': 0.0, 'max_stories': None}}, 'positive finite number'),
            ({'osm': {'scale': 3.0, 'max_stories': 0}}, 'trusted maximum'),
            (
                {
                    'overture': {
                        'sources': {OSM: {'scale': -1.0, 'max_stories': None}}
                    }
                },
                'positive finite number',
            ),
            (
                {
                    'overture': {
                        'sources': {OSM: {'scale_m': 3.0, 'max_stories': None}}
                    }
                },
                'must hold scale and max_stories',
            ),
            (
                {
                    'overture': {
                        'scale': 3.0,
                        'max_stories': None,
                        'sources': {OSM: {'scale': 3.0, 'max_stories': None}},
                    }
                },
                'both sources and a scale',
            ),
            ({'overture': {}}, 'holding scale and max_stories or sources'),
            ({'osm': {'scale': 3.0}}, r'holding scale and max_stories\.'),
            ({'osm': 3.0}, r'holding scale and max_stories\.'),
            ({'overture': {'sources': [OSM]}}, 'must be a mapping from source'),
            (
                {'osm': {'sources': {OSM: {'scale': 3.0, 'max_stories': None}}}},
                "Only the 'overture' entry",
            ),
            (
                {'osm': {'scale': 3.0, 'max_stories': None, 'sources': {}}},
                "Only the 'overture' entry",
            ),
        ],
    )
    def test_invalid_entries_raise(
        self, providers: dict[str, Any], match: str
    ) -> None:
        """A bad scale or maximum, both forms or neither, or misplaced sources raise."""
        with pytest.raises(ValueError, match=match):
            StoriesFromHeightInferer(
                scales={'length_unit': 'm', 'providers': providers}
            )

    @pytest.mark.parametrize(
        'scales',
        [
            {'rule': 'x', 'length_unit': 'm', 'providers': {}},
            {'providers': {}},
            {'length_unit': 'm', 'sources': {}},
            {'length_unit': 'm', 'providers': ['overture']},
        ],
    )
    def test_invalid_top_level_raises(self, scales: dict[str, Any]) -> None:
        """Scales without exactly a unit and a providers mapping raise."""
        with pytest.raises(ValueError, match='exactly length_unit'):
            StoriesFromHeightInferer(scales=scales)

    def test_invalid_units_raise(self) -> None:
        """A height unit or a scales unit that is not a length raises."""
        with pytest.raises(ValueError, match='length unit'):
            StoriesFromHeightInferer(length_unit='kg', scales=FIXTURE_SCALES)
        with pytest.raises(ValueError, match='not one of'):
            StoriesFromHeightInferer(scales={'length_unit': 'kg', 'providers': {}})


class TestCheck:
    """Each asset's diagnosis."""

    def test_verdicts(self, build_inventory: Callable[..., AssetInventory]) -> None:
        """A usable height with a scale passes, and the rest say why not."""
        inventory = build_inventory(
            {
                'height': {'height': 9.0, **OSM_HEIGHT},
                'none': {},
                'na': {'height': 'NA', **OSM_HEIGHT},
                'no_provider': {'height': 9.0, 'sources': f'dataset: {OSM}'},
                'no_source': {'height': 9.0, 'footprint_provider': 'overture'},
                'unlisted_provider': {'height': 9.0, 'footprint_provider': 'osm'},
                'unlisted_source': {
                    'height': 9.0,
                    'footprint_provider': 'overture',
                    'sources': 'dataset: USGS',
                },
                'above': {'height': 20.0, **MICROSOFT_HEIGHT},
            }
        )
        verdicts = STORIES.check(inventory)
        assert verdicts['diagnosis'].to_dict() == {
            'height': DIAGNOSIS_OK,
            'none': DIAGNOSIS_INVALID,
            'na': DIAGNOSIS_INVALID,
            'no_provider': DIAGNOSIS_INVALID,
            'no_source': DIAGNOSIS_INVALID,
            'unlisted_provider': DIAGNOSIS_INFEASIBLE,
            'unlisted_source': DIAGNOSIS_INFEASIBLE,
            'above': DIAGNOSIS_INFEASIBLE,
        }
        assert set(verdicts['resolvable_by']) == {()}
        assert verdicts.loc['no_provider', 'detail'] == 'no footprint_provider'
        assert verdicts.loc['no_source', 'detail'] == 'no height source in sources'
        assert "'osm'" in verdicts.loc['unlisted_provider', 'detail']
        assert "'USGS'" in verdicts.loc['unlisted_source', 'detail']
        assert 'trusted maximum' in verdicts.loc['above', 'detail']

    def test_held_value_ignored(
        self, build_inventory: Callable[..., AssetInventory]
    ) -> None:
        """A count under stories_key neither passes an asset nor fails one."""
        inventory = build_inventory(
            {
                'count_only': {STORIES_FROM_HEIGHT_KEY: 4},
                'count_unscaled': {
                    STORIES_FROM_HEIGHT_KEY: 4,
                    'height': 9.0,
                    'footprint_provider': 'osm',
                },
                'count_and_height': {
                    STORIES_FROM_HEIGHT_KEY: 4,
                    'height': 9.0,
                    **OSM_HEIGHT,
                },
            }
        )
        assert STORIES.check(inventory)['diagnosis'].to_dict() == {
            'count_only': DIAGNOSIS_INVALID,
            'count_unscaled': DIAGNOSIS_INFEASIBLE,
            'count_and_height': DIAGNOSIS_OK,
        }

    def test_per_world_held_value_ignored(
        self, build_inventory: Callable[..., AssetInventory]
    ) -> None:
        """A per-world count under stories_key does not change the verdict."""
        inventory = build_inventory(
            {'a': {'height': 9.0, STORIES_FROM_HEIGHT_KEY: [1, 3], **OSM_HEIGHT}},
            n_pw=2,
        )
        assert STORIES.check(inventory).loc['a', 'diagnosis'] == DIAGNOSIS_OK

    def test_per_world_height_unsupported(
        self, build_inventory: Callable[..., AssetInventory]
    ) -> None:
        """A per-world height is unsupported, a held count beside it or not."""
        inventory = build_inventory(
            {
                'height': {'height': [8.0, 12.0], **OSM_HEIGHT},
                'held': {
                    'height': [8.0, 12.0],
                    STORIES_FROM_HEIGHT_KEY: 3,
                    **OSM_HEIGHT,
                },
            },
            n_pw=2,
        )
        assert STORIES.check(inventory)['diagnosis'].to_dict() == {
            'height': DIAGNOSIS_UNSUPPORTED,
            'held': DIAGNOSIS_UNSUPPORTED,
        }

    def test_per_world_provenance_unsupported(
        self, build_inventory: Callable[..., AssetInventory]
    ) -> None:
        """A per-world provider or sources is unsupported, a list of entries included."""
        inventory = build_inventory(
            {
                'provider': {
                    'height': 9.0,
                    'footprint_provider': ['overture', 'overture'],
                    'sources': f'dataset: {OSM}',
                },
                'sources': {
                    'height': 9.0,
                    'footprint_provider': 'overture',
                    'sources': [f'dataset: {OSM}', f'dataset: {MICROSOFT}'],
                },
                'entries': {
                    'height': 9.9,
                    'footprint_provider': 'overture',
                    'sources': HEIGHT_FROM_OSM,
                },
            },
            n_pw=2,
        )
        verdicts = STORIES.check(inventory)
        assert verdicts['diagnosis'].to_dict() == {
            'provider': DIAGNOSIS_UNSUPPORTED,
            'sources': DIAGNOSIS_UNSUPPORTED,
            'entries': DIAGNOSIS_UNSUPPORTED,
        }
        assert (
            verdicts.loc['sources', 'detail'] == 'sources holds one value per world'
        )

    def test_declared_vector_sources(
        self, build_inventory: Callable[..., AssetInventory]
    ) -> None:
        """A list of entries is one value where sources is a vector feature."""
        inventory = build_inventory(
            {
                'a': {
                    'height': 9.9,
                    'footprint_provider': 'overture',
                    'sources': HEIGHT_FROM_OSM,
                }
            },
            vector_features=['sources'],
        )
        assert STORIES.draw(inventory).to_dict('index') == {
            'a': {STORIES_FROM_HEIGHT_KEY: 2}
        }


class TestDrawAndInfer:
    """What draw returns and infer writes."""

    def test_one_column_draw(
        self, build_inventory: Callable[..., AssetInventory]
    ) -> None:
        """Draw returns the height's count of every asset in one column."""
        inventory = build_inventory(
            {
                'a': {'height': 9.0, **OSM_HEIGHT},
                'b': {'height': 13.0, 'NumberOfStories': 5, **OSM_HEIGHT},
            }
        )
        drawn = STORIES.draw(inventory)
        assert list(drawn.columns) == [STORIES_FROM_HEIGHT_KEY]
        assert drawn[STORIES_FROM_HEIGHT_KEY].to_dict() == {'a': 2, 'b': 3}

    def test_draw_refuses_a_failing_asset(
        self, build_inventory: Callable[..., AssetInventory]
    ) -> None:
        """An asset that fails the check raises in draw."""
        with pytest.raises(ValueError, match='check'):
            STORIES.draw(build_inventory({'none': {}}))

    def test_draw_of_no_assets(
        self, build_inventory: Callable[..., AssetInventory]
    ) -> None:
        """An empty inventory gives an empty frame with the one column."""
        drawn = STORIES.draw(build_inventory({}))
        assert drawn.empty
        assert list(drawn.columns) == [STORIES_FROM_HEIGHT_KEY]

    def test_infer_overwrites(
        self, build_inventory: Callable[..., AssetInventory]
    ) -> None:
        """Infer writes every asset's count over what it holds, and nothing else."""
        inventory = build_inventory(
            {
                'bare': {'height': 9.0, **OSM_HEIGHT},
                'held': {'height': 9.0, STORIES_FROM_HEIGHT_KEY: 5, **OSM_HEIGHT},
                'counted': {'height': 13.0, 'NumberOfStories': 7, **OSM_HEIGHT},
            }
        )
        out = STORIES.infer(inventory)
        features = {
            a: out.get_asset_features(a)[1] for a in ('bare', 'held', 'counted')
        }
        assert {a: f[STORIES_FROM_HEIGHT_KEY] for a, f in features.items()} == {
            'bare': 2,
            'held': 2,
            'counted': 3,
        }
        assert features['counted']['NumberOfStories'] == 7
        assert 'NumberOfStories' not in features['bare']
        assert inventory.get_asset_features('held')[1][STORIES_FROM_HEIGHT_KEY] == 5

    def test_infer_overwrites_a_per_world_value(
        self, build_inventory: Callable[..., AssetInventory]
    ) -> None:
        """A per-world value under stories_key is replaced by the one count."""
        inventory = build_inventory(
            {'a': {'height': 9.0, STORIES_FROM_HEIGHT_KEY: [1, 3], **OSM_HEIGHT}},
            n_pw=2,
        )
        out = STORIES.infer(inventory)
        assert out.get_asset_features('a')[1][STORIES_FROM_HEIGHT_KEY] == 2

    def test_infer_names_the_failures(
        self, build_inventory: Callable[..., AssetInventory]
    ) -> None:
        """Infer raises on any failing asset and names it."""
        inventory = build_inventory({'a': {'height': 9.0, **OSM_HEIGHT}, 'bare': {}})
        with pytest.raises(ValueError, match='bare'):
            STORIES.infer(inventory)

    def test_keys(self, build_inventory: Callable[..., AssetInventory]) -> None:
        """The height and story keys are read and written as configured."""
        inventory = build_inventory({'a': {'h': 9.0, **OSM_HEIGHT}})
        floors = StoriesFromHeightInferer(
            height_key='h',
            stories_key='Floors',
            length_unit='m',
            scales=FIXTURE_SCALES,
        )
        features = floors.infer(inventory).get_asset_features('a')[1]
        assert features['Floors'] == 2
        assert STORIES_FROM_HEIGHT_KEY not in features


def test_exported_and_discovered() -> None:
    """The package exports the class and Importer resolves it."""
    assert brails.inferers.StoriesFromHeightInferer is StoriesFromHeightInferer
    assert (
        Importer().get_class('StoriesFromHeightInferer') is StoriesFromHeightInferer
    )
