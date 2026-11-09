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

"""Tests for replacing values in per-source inventories."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils.source_updates import WITHHELD_COLUMNS, apply_to_sources
from tests.compilers.source_fixtures import (
    PRECEDENCE,
    merge_sources,
    source_features,
    sources_for_one_asset,
)
from tests.imputers.donor_fixtures import square

NEW_VALUES = {'OccupancyClass': 'COM1', 'NumberOfStories': 9}


def withheld_rows(*rows: tuple[str, str, Any]) -> pd.DataFrame:
    """
    Build a table of values withheld from asset ``a``.

    Args:
        *rows (tuple[str, str, Any]):
            ``(feature, source, value)`` per withheld value.

    Returns:
        pandas.DataFrame:
            The table, with columns ``WITHHELD_COLUMNS``.
    """
    return pd.DataFrame(
        [
            {'asset_id': 'a', 'feature': f, 'source': s, 'value': v}
            for f, s, v in rows
        ],
        columns=list(WITHHELD_COLUMNS),
    )


NSI_PAIR = withheld_rows(
    ('OccupancyClass', 'nsi', 'IND1'), ('NumberOfStories', 'nsi', 10)
)


class TestApply:
    """Withholding values and writing their replacements."""

    def test_round_trip(self) -> None:
        """NSI's values are withheld, and the new ones merge back as imputed."""
        sources = sources_for_one_asset()
        before = source_features(sources)
        assert merge_sources(sources).inventory['a'].features['SourceString'] == (
            'N-NL----'
        )
        written = apply_to_sources(
            sources, {'a': dict(NEW_VALUES)}, NSI_PAIR, precedence=PRECEDENCE
        )
        assert written['footprint'] is sources['footprint']
        assert written['local'] is sources['local']
        assert written['nsi'].inventory['a'].features == {}
        assert written['imputed'].inventory['a'].features == NEW_VALUES
        assert written['imputed'].inventory['a'].coordinates == (
            sources['footprint'].inventory['a'].coordinates
        )
        assert source_features(sources) == before
        features = merge_sources(written).inventory['a'].features
        assert features['SourceString'] == 'I-IL----'
        assert features['OccupancyClass'] == 'COM1'
        assert features['YearBuilt'] == 1975

    def test_unchanged_assets_are_shared(self) -> None:
        """A changed source copies only the assets it changes."""
        sources = sources_for_one_asset()
        sources['nsi'].add_asset(
            'b', Asset('b', square(40.0, 0.0), {'OccupancyClass': 'COM1'})
        )
        written = apply_to_sources(sources, {'a': dict(NEW_VALUES)}, NSI_PAIR)
        assert written['nsi'] is not sources['nsi']
        assert written['nsi'].inventory['b'] is sources['nsi'].inventory['b']
        assert written['nsi'].inventory['a'] is not sources['nsi'].inventory['a']

    def test_existing_imputed_values_are_kept(self) -> None:
        """An imputed source gains the new values beside its own."""
        sources = sources_for_one_asset()
        imputed = AssetInventory()
        imputed.add_asset('a', Asset('a', square(0.0, 0.0), {'NumberOfUnits': 3}))
        sources['imputed'] = imputed
        written = apply_to_sources(sources, {'a': dict(NEW_VALUES)}, NSI_PAIR)
        assert written['imputed'].inventory['a'].features == {
            'NumberOfUnits': 3,
            **NEW_VALUES,
        }
        assert imputed.inventory['a'].features == {'NumberOfUnits': 3}

    def test_nothing_to_write(self) -> None:
        """Without updates or withheld values every source is returned as is."""
        sources = sources_for_one_asset()
        written = apply_to_sources(sources, {}, withheld_rows())
        assert all(written[label] is sources[label] for label in sources)
        assert 'imputed' not in written


class TestRefusals:
    """What the function refuses."""

    def test_imputed_value_is_never_overwritten(self) -> None:
        """Writing over a value the imputed source holds raises."""
        sources = sources_for_one_asset()
        imputed = AssetInventory()
        imputed.add_asset(
            'a', Asset('a', square(0.0, 0.0), {'OccupancyClass': 'COM2'})
        )
        sources['imputed'] = imputed
        with pytest.raises(ValueError, match='already holds'):
            apply_to_sources(
                sources, {'a': {'OccupancyClass': 'COM1'}}, withheld_rows()
            )

    @pytest.mark.parametrize(
        ('source', 'value', 'match'),
        [
            ('local', 'IND1', 'may not change'),
            ('footprint', 'IND1', 'may not change'),
            ('imputed', 'IND1', 'may not change'),
            ('derived', 'IND1', 'do not hold'),
            ('nsi', 'COM1', 'does not hold'),
        ],
    )
    def test_withheld_rows_are_checked(
        self, source: str, value: str, match: str
    ) -> None:
        """A withheld row must name a changeable source holding the value."""
        sources = sources_for_one_asset()
        sources['imputed'] = AssetInventory()
        with pytest.raises(ValueError, match=match):
            apply_to_sources(
                sources, {}, withheld_rows(('OccupancyClass', source, value))
            )

    def test_several_worlds_raise(self) -> None:
        """A changed source holding several worlds raises."""
        sources = sources_for_one_asset()
        sources['nsi'].n_pw = 2
        with pytest.raises(ValueError, match='possible worlds'):
            apply_to_sources(sources, {'a': dict(NEW_VALUES)}, NSI_PAIR)

    def test_unknown_asset_raises(self) -> None:
        """A new imputed asset needs coordinates from some source."""
        with pytest.raises(ValueError, match='No source holds'):
            apply_to_sources(
                sources_for_one_asset(),
                {'z': {'OccupancyClass': 'COM1'}},
                withheld_rows(),
            )

    def test_a_higher_source_still_holding_the_feature_raises(self) -> None:
        """Under a precedence, a value that would outrank the new one raises."""
        sources = sources_for_one_asset()
        withheld = withheld_rows(('NumberOfStories', 'nsi', 10))
        updates = {'a': {'OccupancyClass': 'COM1', 'NumberOfStories': 9}}
        with pytest.raises(ValueError, match='still holds'):
            apply_to_sources(sources, updates, withheld, precedence=PRECEDENCE)
        with pytest.raises(ValueError, match='lacks'):
            apply_to_sources(sources, updates, withheld, precedence=['local', 'nsi'])
