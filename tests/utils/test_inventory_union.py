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

"""Tests for the join of inventories with disjoint features."""

from __future__ import annotations

from copy import deepcopy

import pytest

from brails.constants import OCCUPANCY_KEY, STORIES_KEY, UNITS_KEY
from brails.types.asset_inventory import AssetInventory
from brails.utils.inventory_union import union_disjoint
from tests.imputers.donor_fixtures import asset, features_of, inventory

OCC = OCCUPANCY_KEY
UNITS = UNITS_KEY
STORIES = STORIES_KEY


class TestUnion:
    """The union holds every asset and every value of its inputs."""

    def test_union(self) -> None:
        """Assets of either input are kept with the values both hold."""
        one = inventory({'b': asset(0, 0, **{OCC: 'RES1', STORIES: None})})
        other = inventory(
            {'b': asset(50, 0, **{STORIES: 2}), 'a': asset(100, 0, **{UNITS: 1})}
        )
        one.n_pw, other.n_pw = 1, 2
        union = union_disjoint(one, other)
        assert list(union.inventory) == ['a', 'b']
        assert features_of(union) == {
            'a': {UNITS: 1},
            'b': {OCC: 'RES1', STORIES: 2},
        }
        assert union.inventory['b'].coordinates == one.inventory['b'].coordinates
        assert union.n_pw == 2

    def test_inputs_unchanged(self) -> None:
        """The inputs are not modified, and the result holds copies."""
        one = inventory({'a': asset(0, 0, **{OCC: 'RES1'})})
        before = deepcopy(one.inventory['a'].features)
        union = union_disjoint(one)
        union.inventory['a'].features[OCC] = 'COM1'
        assert one.inventory['a'].features == before


class TestRefusals:
    """Inputs that do not join raise."""

    def test_overlap_raises(self) -> None:
        """Two inputs holding one feature of one asset raise, naming it."""
        one = inventory({'a': asset(0, 0, **{OCC: 'RES1'})})
        other = deepcopy(one)
        with pytest.raises(ValueError, match="'a'"):
            union_disjoint(one, other)

    def test_world_mismatch_raises(self) -> None:
        """Inputs with different numbers of worlds above one raise."""
        one, other = AssetInventory(), AssetInventory()
        one.n_pw, other.n_pw = 2, 3
        with pytest.raises(ValueError, match='worlds'):
            union_disjoint(one, other)
