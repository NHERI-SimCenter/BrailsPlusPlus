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

"""Tests for InputValidator.is_inventory in brails.utils.input_validator."""

from __future__ import annotations

from typing import Any

import pytest

from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils import InputValidator as ExportedInputValidator
from brails.utils.input_validator import InputValidator


def test_recognizes_an_inventory() -> None:
    """An AssetInventory is recognized."""
    assert InputValidator.is_inventory(AssetInventory()) is True


def test_recognizes_a_populated_inventory() -> None:
    """Recognition does not depend on the inventory being empty."""
    inventory = AssetInventory()
    inventory.add_asset('A1', Asset('A1', [[-122.4, 37.8]]))
    assert InputValidator.is_inventory(inventory) is True


@pytest.mark.parametrize(
    'value',
    [
        'not an inventory',
        None,
        42,
        {},
        [],
        # An Asset is not an inventory of assets.
        Asset('A1', [[-122.4, 37.8]]),
    ],
)
def test_rejects_other_objects(value: Any) -> None:
    """Anything that is not an AssetInventory is rejected."""
    assert InputValidator.is_inventory(value) is False


def test_is_available_from_the_utils_package() -> None:
    """The method is reachable through the documented import path."""
    assert ExportedInputValidator is InputValidator
    assert ExportedInputValidator.is_inventory(AssetInventory()) is True
