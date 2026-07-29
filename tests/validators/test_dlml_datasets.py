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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.

"""Tests for alias resolution in brails.validators.dlml_datasets."""

from __future__ import annotations

import dlml
import pytest

from brails.exceptions import BrailsError
from brails.validators import METHOD_ALIASES, resolve_dataset_id

EQ_DATASET = 'seismic/building/portfolio/Hazus v6.1'
HU_DATASET = 'hurricane/building/portfolio/Hazus v5.1 coupled'


def test_resolves_the_earthquake_alias() -> None:
    """The earthquake alias maps to the Hazus v6.1 dataset."""
    assert resolve_dataset_id('Hazus Earthquake - Buildings') == EQ_DATASET


@pytest.mark.parametrize(
    'alias',
    [
        'Hazus Hurricane',
        'Hazus Hurricane Wind',
        'Hazus Hurricane Wind - Buildings',
        'Hazus Hurricane Wind - Buildings - Coupled',
    ],
)
def test_resolves_the_hurricane_aliases(alias: str) -> None:
    """Every hurricane wind alias maps to the coupled dataset."""
    assert resolve_dataset_id(alias) == HU_DATASET


@pytest.mark.parametrize('dataset_id', [EQ_DATASET, HU_DATASET])
def test_passes_through_dataset_identifiers(dataset_id: str) -> None:
    """A canonical identifier resolves to itself."""
    assert resolve_dataset_id(dataset_id) == dataset_id


def test_rejects_an_unknown_name() -> None:
    """An unknown name is rejected, and the valid aliases are listed."""
    with pytest.raises(BrailsError) as excinfo:
        resolve_dataset_id('Hazus Earthquake')

    message = str(excinfo.value)
    assert 'Hazus Earthquake' in message
    assert 'Hazus Earthquake - Buildings' in message


def test_every_alias_targets_a_real_dataset() -> None:
    """No alias points at a dataset DLML does not publish."""
    published = set(dlml.list_datasets())
    assert set(METHOD_ALIASES.values()) <= published


def test_every_alias_targets_a_dataset_with_a_schema() -> None:
    """An alias to a dataset without a schema would be a false promise."""
    for dataset_id in set(METHOD_ALIASES.values()):
        assert dlml.get_schema(dataset_id)
