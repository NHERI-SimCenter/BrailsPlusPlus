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

"""Tests for brails.validators.schema_validator."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from brails.exceptions import BrailsError
from brails.validators import (
    InventoryValidator,
    SchemaValidator,
    ValidationReport,
)

EQ_DATASET = 'seismic/building/portfolio/Hazus v6.1'
NO_SCHEMA_DATASET = 'flood/building/portfolio/Hazus v6.1'

if TYPE_CHECKING:
    from collections.abc import Callable

    from brails.types.asset_inventory import AssetInventory


@pytest.fixture
def validator() -> SchemaValidator:
    """Return a validator for the Hazus earthquake building dataset."""
    return SchemaValidator('Hazus Earthquake - Buildings')


# ---------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------


def test_accepts_an_alias_and_an_identifier() -> None:
    """Both spellings select the same dataset."""
    assert SchemaValidator('Hazus Earthquake - Buildings').dataset_id == (
        SchemaValidator(EQ_DATASET).dataset_id
    )


def test_rejects_a_dataset_without_a_schema() -> None:
    """A dataset with no input schema cannot be validated against."""
    with pytest.raises(BrailsError) as excinfo:
        SchemaValidator(NO_SCHEMA_DATASET)

    message = str(excinfo.value)
    assert NO_SCHEMA_DATASET in message
    # The error points at what can be used instead.
    assert EQ_DATASET in message


def test_available_datasets_lists_only_those_with_schemas() -> None:
    """Every listed dataset can actually be constructed."""
    available = SchemaValidator.available_datasets()

    assert EQ_DATASET in available
    assert NO_SCHEMA_DATASET not in available
    for dataset_id in available:
        assert SchemaValidator(dataset_id).dataset_id == dataset_id


def test_is_an_inventory_validator(validator: SchemaValidator) -> None:
    """The concrete validator implements the subsystem's base class."""
    assert isinstance(validator, InventoryValidator)


# ---------------------------------------------------------------------
# the base class contract
# ---------------------------------------------------------------------


def test_the_base_class_cannot_be_instantiated() -> None:
    """The base class declares an interface it does not implement."""
    with pytest.raises(TypeError):
        InventoryValidator()


def test_a_subclass_must_implement_validate() -> None:
    """A subclass that does not validate is still abstract."""

    class Incomplete(InventoryValidator):
        """A validator that forgot to implement validate."""

    with pytest.raises(TypeError):
        Incomplete()


# ---------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------


def test_accepts_a_conformant_inventory(
    build_inventory: Callable[..., AssetInventory], validator: SchemaValidator
) -> None:
    """An inventory that satisfies the schema yields an empty report."""
    inventory = build_inventory(
        {
            'A1': {
                'StructureType': 'W1',
                'DesignLevel': 'High-Code',
                'OccupancyClass': 'RES1',
            },
            'A2': {
                'StructureType': 'C1',
                'DesignLevel': 'Low-Code',
                'HeightClass': 'Mid-Rise',
                'OccupancyClass': 'COM1',
            },
        }
    )

    report = validator.validate(inventory)

    assert report.is_valid is True
    assert report.n_assets == 2


def test_reports_a_missing_height_class(
    build_inventory: Callable[..., AssetInventory], validator: SchemaValidator
) -> None:
    """C1 requires a height class, so omitting it is a violation."""
    inventory = build_inventory(
        {'A1': {'StructureType': 'C1', 'DesignLevel': 'Low-Code'}}
    )

    report = validator.validate(inventory)

    assert report.is_valid is False
    assert 'HeightClass' in ' '.join(report.by_asset()['A1'])


def test_reports_an_invalid_enum_value(
    build_inventory: Callable[..., AssetInventory], validator: SchemaValidator
) -> None:
    """A value outside the schema's enumeration is a violation."""
    inventory = build_inventory(
        {'A1': {'StructureType': 'NotAType', 'DesignLevel': 'Low-Code'}}
    )

    report = validator.validate(inventory)

    assert 'StructureType' in ' '.join(report.by_asset()['A1'])


def test_reports_a_conditional_violation(
    build_inventory: Callable[..., AssetInventory], validator: SchemaValidator
) -> None:
    """URM caps the design level and the height class."""
    inventory = build_inventory(
        {
            'A1': {
                'StructureType': 'URM',
                'DesignLevel': 'High-Code',
                'HeightClass': 'High-Rise',
            }
        }
    )

    messages = ' '.join(validator.validate(inventory).by_asset()['A1'])

    assert 'DesignLevel' in messages
    assert 'HeightClass' in messages


def test_raw_nsi_attributes_are_mass_nonconformant(
    build_inventory: Callable[..., AssetInventory],
    validator: SchemaValidator,
) -> None:
    """An inventory carrying only raw attributes fails for every asset."""
    inventory = build_inventory(
        {
            f'A{index}': {
                'OccupancyClass': 'RES1',
                'NumberOfStories': 2,
                'YearBuilt': 1975,
            }
            for index in range(5)
        }
    )

    report = validator.validate(inventory)

    assert len(report) == 5
    for messages in report.by_asset().values():
        assert 'StructureType' in ' '.join(messages)


def test_does_not_modify_the_inventory(
    build_inventory: Callable[..., AssetInventory], validator: SchemaValidator
) -> None:
    """Validation is read-only."""
    inventory = build_inventory(
        {'A1': {'StructureType': 'URM', 'DesignLevel': 'High-Code'}}
    )
    before = deepcopy(inventory.inventory['A1'].features)

    validator.validate(inventory)

    assert inventory.inventory['A1'].features == before


def test_rejects_a_non_inventory(validator: SchemaValidator) -> None:
    """Anything that is not an AssetInventory is rejected."""
    with pytest.raises(TypeError):
        validator.validate('not an inventory')


# ---------------------------------------------------------------------
# possible worlds
# ---------------------------------------------------------------------


def test_validates_every_world(
    build_inventory: Callable[..., AssetInventory], validator: SchemaValidator
) -> None:
    """A violation in one world alone is reported against that world."""
    inventory = build_inventory(
        {
            'A1': {
                # W1 needs no height class; URM caps the design level and
                # requires one, so only the second world is invalid.
                'StructureType': ['W1', 'URM'],
                'DesignLevel': 'High-Code',
                'OccupancyClass': 'RES1',
            }
        },
        n_pw=2,
    )

    report = validator.validate(inventory)

    assert report.n_pw == 2
    assert report.is_valid is False
    assert set(report.by_world()) == {1}
    assert 'DesignLevel' in ' '.join(report.by_world()[1]['A1'])


def test_reports_a_violation_present_in_every_world(
    build_inventory: Callable[..., AssetInventory],
    validator: SchemaValidator,
) -> None:
    """A violation in all worlds is recorded against each of them."""
    inventory = build_inventory(
        {'A1': {'StructureType': ['URM', 'URM'], 'DesignLevel': 'High-Code'}},
        n_pw=2,
    )

    report = validator.validate(inventory)

    assert set(report.by_world()) == {0, 1}
    # Collapsed, the repeated message appears once.
    assert len(report.by_asset()['A1']) == len(report.by_world()[0]['A1'])


def test_rejects_ragged_realizations(
    build_inventory: Callable[..., AssetInventory], validator: SchemaValidator
) -> None:
    """Features must agree with the inventory's number of worlds."""
    inventory = build_inventory(
        {'A1': {'StructureType': ['W1', 'URM', 'C1']}}, n_pw=2
    )

    with pytest.raises(BrailsError, match='StructureType'):
        validator.validate(inventory)


def test_returns_a_validation_report(
    build_inventory: Callable[..., AssetInventory], validator: SchemaValidator
) -> None:
    """The documented return type is what comes back."""
    report = validator.validate(build_inventory({}))

    assert isinstance(report, ValidationReport)
    assert report.dataset_id == EQ_DATASET
