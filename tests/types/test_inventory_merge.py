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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.

"""Tests for AssetInventory.merge and the source string it writes."""

from __future__ import annotations

import copy
import itertools
import math
import re
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from brails.exceptions import BrailsError
from brails.types.asset_inventory import (
    HAZUS_EQ_SOURCE_FEATURES,
    MAX_REPORTED_WORLD_PROBLEMS,
    SOURCE_CODE_NONE,
    SOURCE_CODES,
    SOURCE_STRING_FEATURE,
    AssetInventory,
)
from brails.types.housing_unit_inventory import HousingUnitInventory

if TYPE_CHECKING:
    from collections.abc import Callable

    Build = Callable[..., AssetInventory]

# Precedence for the three standard sources.
PRECEDENCE = ['local', 'nsi', 'footprint']

COORDINATES = {
    'g1': [[-122.41, 37.77], [-122.40, 37.77], [-122.40, 37.78], [-122.41, 37.77]],
    'g2': [[-122.42, 37.76], [-122.41, 37.76], [-122.41, 37.77], [-122.42, 37.76]],
    'g3': [[-122.43, 37.75], [-122.42, 37.75], [-122.42, 37.76], [-122.43, 37.75]],
}

FOOTPRINT_FEATURES = {
    'g1': {'footprint_provider': 'overture', 'NumberOfStories': 'NA'},
    'g2': {'footprint_provider': 'overture', 'NumberOfStories': 3},
    'g3': {'footprint_provider': 'overture', 'NumberOfStories': None},
}

NSI_FEATURES = {
    'g1': {
        'OccupancyClass': 'RES1',
        'NumberOfStories': 2,
        'YearBuilt': 1925,
        'ReplacementCost': 410000.0,
    },
    'g2': {
        'OccupancyClass': 'COM1',
        'NumberOfStories': 4,
        'YearBuilt': float('nan'),
        'ReplacementCost': 1250000.0,
    },
}

LOCAL_FEATURES = {
    'g1': {'YearBuilt': 1931},
    'g3': {'OccupancyClass': 'RES3A', 'YearBuilt': None},
}


def three_sources(
    build: Build, order: list[str] | None = None
) -> tuple[AssetInventory, dict[str, AssetInventory]]:
    """Return the footprint spine and the NSI and local sources."""
    order = order or ['g1', 'g2', 'g3']

    def in_order(features: dict[str, dict[str, Any]]) -> AssetInventory:
        inventory = build({a: features[a] for a in order if a in features})
        for asset_id, asset in inventory.inventory.items():
            asset.coordinates = copy.deepcopy(COORDINATES[asset_id])
        return inventory

    return in_order(FOOTPRINT_FEATURES), {
        'nsi': in_order(NSI_FEATURES),
        'local': in_order(LOCAL_FEATURES),
    }


def snapshot(inventory: AssetInventory) -> list[tuple[Any, ...]]:
    """Represent an inventory as a list, keeping its asset order."""
    return [
        (repr(asset_id), repr(asset.coordinates), repr(asset.features))
        for asset_id, asset in inventory.inventory.items()
    ] + [
        ('n_pw', inventory.n_pw),
        ('vector_features', tuple(inventory.vector_features)),
        ('housing_units', inventory.housing_unit_inventory is None),
    ]


def merge_three(build: Build, **kwargs: Any) -> AssetInventory:
    """Merge the three standard sources."""
    spine, others = three_sources(build)
    return spine.merge(others, PRECEDENCE, name='footprint', **kwargs)


class TestThreeSourceMerge:
    """Precedence, missing values and geometry in a three-source merge."""

    def test_each_feature_comes_from_the_first_source_holding_it(
        self, build_inventory: Build
    ) -> None:
        """Local beats NSI beats footprint, feature by feature."""
        merged = merge_three(build_inventory)

        g1 = merged.inventory['g1'].features
        assert g1['YearBuilt'] == 1931  # local over NSI
        assert g1['NumberOfStories'] == 2  # NSI over the footprint's 'NA'
        assert g1['OccupancyClass'] == 'RES1'
        assert g1['footprint_provider'] == 'overture'

        g2 = merged.inventory['g2'].features
        assert g2['NumberOfStories'] == 4  # NSI over a footprint value
        assert g2['OccupancyClass'] == 'COM1'

        g3 = merged.inventory['g3'].features
        assert g3['OccupancyClass'] == 'RES3A'

    def test_the_full_source_order_is_honored(self, build_inventory: Build) -> None:
        """Local, NSI, imputed and drawn values win in that order."""
        spine = build_inventory({'g1': {}})
        local = build_inventory({'g1': {'YearBuilt': 1931}})
        nsi = build_inventory({'g1': {'YearBuilt': 1925, 'NumberOfStories': 2}})
        imputed = build_inventory(
            {'g1': {'YearBuilt': 1940, 'NumberOfStories': 3, 'DesignLevel': 'MC'}}
        )
        drawn = build_inventory(
            {
                'g1': {
                    'YearBuilt': 1950,
                    'NumberOfStories': 4,
                    'DesignLevel': 'LC',
                    'StructureType': 'W1',
                }
            }
        )

        merged = spine.merge(
            {'local': local, 'nsi': nsi, 'imputed': imputed, 'drawn': drawn},
            ['local', 'nsi', 'imputed', 'drawn', 'footprint'],
            name='footprint',
        )

        g1 = merged.inventory['g1'].features
        assert (
            g1['YearBuilt'],
            g1['NumberOfStories'],
            g1['DesignLevel'],
            g1['StructureType'],
        ) == (1931, 2, 'MC', 'W1')
        assert g1[SOURCE_STRING_FEATURE] == '--NLID--'

    @pytest.mark.parametrize('marker', [None, float('nan'), pd.NA, pd.NaT, 'NA', ''])
    def test_missing_values_never_win(
        self, build_inventory: Build, marker: Any
    ) -> None:
        """A missing value gives way to a lower source holding a value."""
        spine = build_inventory({'g1': {'YearBuilt': 1950}})
        local = build_inventory({'g1': {'YearBuilt': marker}})

        merged = spine.merge(
            {'local': local}, ['local', 'footprint'], name='footprint'
        )

        g1 = merged.inventory['g1'].features
        assert g1['YearBuilt'] == 1950
        assert g1[SOURCE_STRING_FEATURE] == '---F----'

    def test_zero_is_a_value_and_the_empty_string_is_missing(
        self, build_inventory: Build
    ) -> None:
        """A zero wins. An empty string, as a blank CSV cell reads, does not."""
        spine = build_inventory({'g1': {'NumberOfUnits': 4, 'DesignLevel': 'HC'}})
        local = build_inventory({'g1': {'NumberOfUnits': 0, 'DesignLevel': ''}})

        merged = spine.merge(
            {'local': local}, ['local', 'footprint'], name='footprint'
        )

        g1 = merged.inventory['g1'].features
        assert g1['NumberOfUnits'] == 0
        assert g1['DesignLevel'] == 'HC'
        assert g1[SOURCE_STRING_FEATURE] == '-L--F---'

    def test_no_value_anywhere_leaves_the_marker(
        self, build_inventory: Build
    ) -> None:
        """No value anywhere leaves the marker and marks no source in the string."""
        merged = merge_three(build_inventory)

        # g2: NSI's NaN year and no local record leave no year at all.
        assert math.isnan(merged.inventory['g2'].features['YearBuilt'])
        # g3: YearBuilt None in local, NumberOfStories None in the footprint.
        assert merged.inventory['g3'].features['YearBuilt'] is None
        assert merged.inventory['g3'].features['NumberOfStories'] is None
        assert merged.inventory['g3'].features[SOURCE_STRING_FEATURE][2:4] == '--'

    def test_a_feature_nobody_holds_keeps_the_first_marker(
        self, build_inventory: Build
    ) -> None:
        """The marker comes from the highest source that has the key."""
        spine = build_inventory({'g1': {'NumberOfStories': 'NA'}})
        nsi = build_inventory({'g1': {'NumberOfStories': None}})

        merged = spine.merge({'nsi': nsi}, ['nsi', 'footprint'], name='footprint')
        assert merged.inventory['g1'].features['NumberOfStories'] is None

        merged = spine.merge({'nsi': nsi}, ['footprint', 'nsi'], name='footprint')
        assert merged.inventory['g1'].features['NumberOfStories'] == 'NA'

    def test_spine_defines_assets_and_geometry(self, build_inventory: Build) -> None:
        """Assets missing from a source still appear, with spine geometry."""
        spine, others = three_sources(build_inventory)
        others['nsi'].inventory['g1'].coordinates = [[0.0, 0.0]]

        merged = spine.merge(others, PRECEDENCE, name='footprint')

        assert list(merged.inventory) == ['g1', 'g2', 'g3']
        assert merged.inventory['g1'].coordinates == COORDINATES['g1']
        assert merged.housing_unit_inventory is None

    def test_spine_alone(self, build_inventory: Build) -> None:
        """A merge with no other source copies the spine and writes its string."""
        spine, _ = three_sources(build_inventory)
        merged = spine.merge({}, ['footprint'], name='footprint')
        assert merged.inventory['g2'].features['NumberOfStories'] == 3
        assert merged.inventory['g2'].features[SOURCE_STRING_FEATURE] == '--F-----'

    def test_vector_features_are_the_union(self, build_inventory: Build) -> None:
        """The result declares every source's vector features once."""
        spine = build_inventory({'g1': {}}, vector_features=['Openings'])
        nsi = build_inventory({'g1': {}}, vector_features=['HousingUnits'])

        merged = spine.merge({'nsi': nsi}, ['nsi', 'footprint'], name='footprint')

        assert merged.vector_features == ['HousingUnits', 'Openings']


class TestOrderIndependence:
    """The result does not depend on the order of sources or assets."""

    def test_order_of_others_and_of_assets_does_not_matter(
        self, build_inventory: Build
    ) -> None:
        """Every permutation of others and of asset insertion agrees."""
        reference = snapshot(merge_three(build_inventory))

        for asset_order in itertools.permutations(['g1', 'g2', 'g3']):
            spine, others = three_sources(build_inventory, list(asset_order))
            for labels in itertools.permutations(others):
                reordered = {label: others[label] for label in labels}
                merged = spine.merge(reordered, PRECEDENCE, name='footprint')
                assert snapshot(merged) == reference

    def test_assets_are_inserted_in_ascending_string_order(
        self, build_inventory: Build
    ) -> None:
        """Integer and string ids sort by their string form."""
        inventory = build_inventory({10: {}, 'b': {}, 2: {}, 'a': {}})

        merged = inventory.merge({}, ['footprint'], name='footprint')

        assert list(merged.inventory) == [10, 2, 'a', 'b']


class TestSourceString:
    """The characters, their order and their meaning."""

    def test_default_features_and_their_order(self, build_inventory: Build) -> None:
        """The declared order is fixed and the string has one character each."""
        assert HAZUS_EQ_SOURCE_FEATURES == (
            'OccupancyClass',
            'NumberOfUnits',
            'NumberOfStories',
            'YearBuilt',
            'DesignLevel',
            'StructureType',
            'HeightClass',
            'ReplacementCost',
        )
        merged = merge_three(build_inventory)
        strings = {
            asset_id: asset.features[SOURCE_STRING_FEATURE]
            for asset_id, asset in merged.inventory.items()
        }
        assert strings == {
            'g1': 'N-NL---N',
            'g2': 'N-N----N',
            'g3': 'L-------',
        }

    def test_character_order_follows_source_features(
        self, build_inventory: Build
    ) -> None:
        """Reversing the declared features reverses the string."""
        forward = ['OccupancyClass', 'NumberOfStories', 'YearBuilt']
        backward = forward[::-1]

        g1_forward = merge_three(build_inventory, source_features=forward)
        g1_backward = merge_three(build_inventory, source_features=backward)

        assert g1_forward.inventory['g1'].features[SOURCE_STRING_FEATURE] == 'NNL'
        assert g1_backward.inventory['g1'].features[SOURCE_STRING_FEATURE] == 'LNN'

    def test_an_iterator_of_features_is_read_once(
        self, build_inventory: Build
    ) -> None:
        """An iterator of features is read once, by the checks and the string."""
        forward = ['OccupancyClass', 'NumberOfStories', 'YearBuilt']

        merged = merge_three(build_inventory, source_features=iter(forward))

        assert merged.inventory['g1'].features[SOURCE_STRING_FEATURE] == 'NNL'

    def test_only_codes_and_the_none_marker(self, build_inventory: Build) -> None:
        """Every character is an uppercase source code or the none marker."""
        allowed = set(SOURCE_CODES.values()) | {SOURCE_CODE_NONE}
        for asset in merge_three(build_inventory).inventory.values():
            string = asset.features[SOURCE_STRING_FEATURE]
            assert len(string) == len(HAZUS_EQ_SOURCE_FEATURES)
            assert set(string) <= allowed
            assert re.fullmatch(r'[A-Z-]+', string)

    @pytest.mark.parametrize(
        ('label', 'code'),
        [
            ('local', 'L'),
            ('footprint', 'F'),
            ('nsi', 'N'),
            ('derived', 'R'),
            ('imputed', 'I'),
            ('drawn', 'D'),
        ],
    )
    def test_each_label_writes_its_code(
        self, build_inventory: Build, label: str, code: str
    ) -> None:
        """A value held only by one source is marked with that source's code."""
        assert SOURCE_CODES[label] == code
        spine_label = 'local' if label == 'footprint' else 'footprint'
        spine = build_inventory({'g1': {}})
        source = build_inventory({'g1': {'YearBuilt': 1950}})

        merged = spine.merge(
            {label: source},
            [spine_label, label],
            name=spine_label,
            source_features=['YearBuilt', 'DesignLevel'],
        )

        assert merged.inventory['g1'].features[SOURCE_STRING_FEATURE] == (
            code + SOURCE_CODE_NONE
        )

    def test_no_source_string_when_features_are_none(
        self, build_inventory: Build
    ) -> None:
        """source_features=None writes no source string."""
        merged = merge_three(build_inventory, source_features=None)
        assert all(
            SOURCE_STRING_FEATURE not in asset.features
            for asset in merged.inventory.values()
        )

    def test_value_does_not_matter_only_who_holds_it(
        self, build_inventory: Build
    ) -> None:
        """Two different values from the same source give the same character."""
        spine = build_inventory({'g1': {}, 'g2': {}})
        nsi = build_inventory({'g1': {'YearBuilt': 1900}, 'g2': {'YearBuilt': 2001}})

        merged = spine.merge({'nsi': nsi}, ['nsi', 'footprint'], name='footprint')

        assert (
            merged.inventory['g1'].features[SOURCE_STRING_FEATURE]
            == merged.inventory['g2'].features[SOURCE_STRING_FEATURE]
            == '---N----'
        )


class TestWorldInvariance:
    """Per-world values and their counts."""

    def test_per_world_list_and_scalar_give_the_same_string(
        self, build_inventory: Build
    ) -> None:
        """A list of per-world values is marked like a single value."""
        spine = build_inventory({'g1': {}, 'g2': {}})
        drawn = build_inventory(
            {
                'g1': {'YearBuilt': [1921, 1934, 1950]},
                'g2': {'YearBuilt': 1934},
            },
            n_pw=3,
        )

        merged = spine.merge(
            {'drawn': drawn}, ['footprint', 'drawn'], name='footprint'
        )

        g1 = merged.inventory['g1'].features
        g2 = merged.inventory['g2'].features
        assert g1['YearBuilt'] == [1921, 1934, 1950]
        assert g1[SOURCE_STRING_FEATURE] == g2[SOURCE_STRING_FEATURE] == '---D----'
        assert merged.n_pw == 3

    def test_each_world_merged_alone_gives_the_same_string(
        self, build_inventory: Build
    ) -> None:
        """One world at a time gives the same values and string, no list missing."""
        spine = build_inventory({'g1': {'NumberOfStories': 2}})
        drawn = build_inventory(
            {'g1': {'YearBuilt': [1921, 1934, 1950], 'DesignLevel': 'LC'}},
            n_pw=3,
        )
        precedence = ['footprint', 'drawn']

        merged = spine.merge({'drawn': drawn}, precedence, name='footprint')
        g1 = merged.inventory['g1'].features

        for world in range(drawn.n_pw):
            single = spine.merge(
                {'drawn': drawn.get_world_realization(world)},
                precedence,
                name='footprint',
            )
            features = single.inventory['g1'].features
            assert features['YearBuilt'] == g1['YearBuilt'][world]
            assert features[SOURCE_STRING_FEATURE] == g1[SOURCE_STRING_FEATURE]

    def test_a_list_counts_as_a_value_whatever_it_holds(
        self, build_inventory: Build
    ) -> None:
        """Missing values inside a list do not change the winner."""
        spine = build_inventory({'g1': {'YearBuilt': 1900}})
        drawn = build_inventory({'g1': {'YearBuilt': [None, float('nan')]}}, n_pw=2)

        merged = spine.merge(
            {'drawn': drawn}, ['drawn', 'footprint'], name='footprint'
        )

        assert merged.inventory['g1'].features[SOURCE_STRING_FEATURE] == '---D----'

    def test_one_world_sources_mix_with_many(self, build_inventory: Build) -> None:
        """Sources with one world merge with sources with several."""
        spine = build_inventory({'g1': {}})
        imputed = build_inventory({'g1': {'NumberOfStories': [1, 2]}}, n_pw=2)
        drawn = build_inventory({'g1': {'YearBuilt': [1950, 1960]}}, n_pw=2)

        merged = spine.merge(
            {'imputed': imputed, 'drawn': drawn},
            ['footprint', 'imputed', 'drawn'],
            name='footprint',
        )

        assert merged.n_pw == 2
        merged.validate_possible_worlds()

    def test_different_world_counts_raise(self, build_inventory: Build) -> None:
        """Two sources with different counts above one cannot be merged."""
        spine = build_inventory({'g1': {}}, n_pw=2)
        drawn = build_inventory({'g1': {'YearBuilt': [1950, 1960, 1970]}}, n_pw=3)

        with pytest.raises(ValueError, match='different numbers of possible'):
            spine.merge({'drawn': drawn}, ['footprint', 'drawn'], name='footprint')

    def test_a_one_world_list_beside_several_worlds_raises(
        self, build_inventory: Build
    ) -> None:
        """A valid one-world source holding a list is refused by its label."""
        spine = build_inventory({'g1': {'Tags': ['a']}})
        spine.validate_possible_worlds()
        drawn = build_inventory({'g1': {'YearBuilt': [1950, 1960]}}, n_pw=2)

        with pytest.raises(BrailsError, match="Source 'footprint'.*'Tags'"):
            spine.merge({'drawn': drawn}, ['footprint', 'drawn'], name='footprint')


class TestInputsUnchanged:
    """Neither the spine nor any source is modified."""

    def test_inputs_are_unmutated(self, build_inventory: Build) -> None:
        """Snapshots taken before and after the merge agree."""
        spine, others = three_sources(build_inventory)
        before = [snapshot(spine)] + [snapshot(inv) for inv in others.values()]

        spine.merge(others, PRECEDENCE, name='footprint')

        after = [snapshot(spine)] + [snapshot(inv) for inv in others.values()]
        assert after == before

    def test_result_shares_no_mutable_value(self, build_inventory: Build) -> None:
        """Changing the result leaves the inputs as they were."""
        spine = build_inventory({'g1': {'Tags': ['a']}}, vector_features=['Tags'])
        spine.inventory['g1'].coordinates = copy.deepcopy(COORDINATES['g1'])
        drawn = build_inventory({'g1': {'YearBuilt': [1950, 1960]}}, n_pw=2)

        merged = spine.merge(
            {'drawn': drawn}, ['footprint', 'drawn'], name='footprint'
        )
        merged.inventory['g1'].features['Tags'].append('b')
        merged.inventory['g1'].features['YearBuilt'][0] = 0
        merged.inventory['g1'].coordinates[0][0] = 0.0

        assert spine.inventory['g1'].features['Tags'] == ['a']
        assert drawn.inventory['g1'].features['YearBuilt'] == [1950, 1960]
        assert spine.inventory['g1'].coordinates == COORDINATES['g1']
        assert merged.inventory['g1'] is not spine.inventory['g1']


class TestRefusals:
    """Inputs the merge refuses, and the messages that name them."""

    @pytest.mark.parametrize(
        ('precedence', 'match'),
        [
            (['local', 'nsi'], r"omitted: \['footprint'\]"),
            (['local', 'nsi', 'footprint', 'nsi'], r"Repeated: \['nsi'\]"),
            (['local', 'nsi', 'footprint', 'drawn'], r"not a source: \['drawn'\]"),
        ],
    )
    def test_precedence_must_list_every_label_once(
        self, build_inventory: Build, precedence: list[str], match: str
    ) -> None:
        """A missing, repeated or unknown label raises."""
        spine, others = three_sources(build_inventory)
        with pytest.raises(ValueError, match=match):
            spine.merge(others, precedence, name='footprint')

    def test_precedence_as_a_string_raises(self, build_inventory: Build) -> None:
        """A single label is not a precedence list."""
        spine, _ = three_sources(build_inventory)
        with pytest.raises(TypeError, match="not the string 'footprint'"):
            spine.merge({}, 'footprint', name='footprint')

    def test_spine_label_reused_by_another_source_raises(
        self, build_inventory: Build
    ) -> None:
        """The spine label may not also label one of others."""
        spine, others = three_sources(build_inventory)
        with pytest.raises(ValueError, match="'nsi' names this inventory"):
            spine.merge(others, PRECEDENCE, name='nsi')

    def test_others_that_is_not_a_mapping_raises(
        self, build_inventory: Build
    ) -> None:
        """The others argument must map labels to inventories."""
        spine, _ = three_sources(build_inventory)
        with pytest.raises(TypeError, match='inventories, not list'):
            spine.merge([], ['footprint'], name='footprint')

    def test_a_source_that_is_not_an_inventory_raises(
        self, build_inventory: Build
    ) -> None:
        """Every source must be an AssetInventory."""
        spine, _ = three_sources(build_inventory)
        with pytest.raises(
            TypeError, match=r"'nsi' is not an AssetInventory \(type dict\)"
        ):
            spine.merge({'nsi': {}}, ['nsi', 'footprint'], name='footprint')

    def test_label_without_code_raises_only_for_a_source_string(
        self, build_inventory: Build
    ) -> None:
        """An unknown label is refused when a source string is requested."""
        spine, others = three_sources(build_inventory)
        others['assessor'] = others.pop('local')
        precedence = ['assessor', 'nsi', 'footprint']

        with pytest.raises(ValueError, match=r"labels \['assessor'\]"):
            spine.merge(others, precedence, name='footprint')

        merged = spine.merge(
            others, precedence, name='footprint', source_features=None
        )
        assert merged.inventory['g1'].features['YearBuilt'] == 1931

    @pytest.mark.parametrize(
        ('source_features', 'error', 'match'),
        [
            (['YearBuilt', 'YearBuilt'], ValueError, 'repeats'),
            ([SOURCE_STRING_FEATURE], ValueError, 'cannot include'),
            ('YearBuilt', TypeError, "not the string 'YearBuilt'"),
        ],
    )
    def test_bad_source_features_raise(
        self,
        build_inventory: Build,
        source_features: Any,
        error: type[Exception],
        match: str,
    ) -> None:
        """Repeated or self-referring features, or a single string, raise."""
        with pytest.raises(error, match=match):
            merge_three(build_inventory, source_features=source_features)

    def test_ids_absent_from_the_spine_raise_and_are_counted(
        self, build_inventory: Build
    ) -> None:
        """Foreign ids are named up to the reporting limit and the rest counted."""
        extra = MAX_REPORTED_WORLD_PROBLEMS + 3
        spine = build_inventory({'g1': {}})
        nsi = build_inventory(
            {'g1': {}} | {f'fd{number:02d}': {} for number in range(extra)}
        )

        with pytest.raises(ValueError, match='absent') as caught:
            spine.merge({'nsi': nsi}, ['nsi', 'footprint'], name='footprint')

        message = str(caught.value)
        assert message.startswith(f'{extra} asset id(s)')
        assert "'fd00' (nsi)" in message
        assert "'fd09' (nsi)" in message
        assert "'fd10'" not in message
        assert '... and 3 more' in message

    def test_a_carried_source_string_raises(self, build_inventory: Build) -> None:
        """A source that already carries a source string is refused."""
        first = merge_three(build_inventory)
        nsi = build_inventory({'g1': {'NumberOfUnits': 1}})

        with pytest.raises(ValueError, match='already carry'):
            first.merge({'nsi': nsi}, ['nsi', 'local'], name='local')

    def test_a_housing_unit_inventory_raises(self, build_inventory: Build) -> None:
        """A source holding a housing unit inventory is refused."""
        spine, others = three_sources(build_inventory)
        others['nsi'].housing_unit_inventory = HousingUnitInventory()

        with pytest.raises(ValueError, match=r"\['nsi'\] hold a housing unit"):
            spine.merge(others, PRECEDENCE, name='footprint')
