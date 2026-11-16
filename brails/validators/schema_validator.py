# Copyright (c) 2024 The Regents of the University of California
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

"""
This module validates inventories against DLML input schemas.

.. autosummary::

    SchemaValidator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dlml

from brails.exceptions import BrailsError
from brails.types.asset_inventory import is_probabilistic_feature
from brails.utils.input_validator import InputValidator
from brails.validators.dlml_datasets import resolve_dataset_id
from brails.validators.inventory_validator import InventoryValidator
from brails.validators.validation_report import ValidationReport

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory


class SchemaValidator(InventoryValidator):
    """
    Validate an AssetInventory against a DLML input schema.

    The Damage and Loss Model Library publishes, for each method that
    needs one, a JSON Schema describing what that method requires of an
    asset. This class checks an inventory against one of those schemas.
    One class therefore serves every method DLML covers; the dataset is
    chosen per instance.

    A green report means the inventory satisfies that schema. It is not a
    statement that the inventory is otherwise complete or plausible.

    To import the :class:`SchemaValidator` class, use:

    .. code-block:: python

        from brails.validators import SchemaValidator

    Examples:
        >>> from brails.validators import SchemaValidator
        >>> validator = SchemaValidator('Hazus Earthquake - Buildings')
        >>> validator.dataset_id
        'seismic/building/portfolio/Hazus v6.1'
    """

    def __init__(self, dataset: str) -> None:
        """
        Initialize the validator for one DLML dataset.

        Args:
            dataset (str):
                A method alias such as ``'Hazus Earthquake - Buildings'``
                or a DLML dataset identifier such as
                ``'seismic/building/portfolio/Hazus v6.1'``.

        Raises:
            BrailsError:
                If ``dataset`` is not a known alias or dataset
                identifier, or if the dataset ships no input schema.
        """
        self.dataset_id = resolve_dataset_id(dataset)

        try:
            self._schema = dlml.get_schema(self.dataset_id)
        except Exception as exc:
            available = ', '.join(repr(name) for name in self.available_datasets())
            raise BrailsError(
                f'The DLML dataset {self.dataset_id!r} does not provide an '
                f'input schema, so an inventory cannot be validated against '
                f'it. Datasets that currently provide one: {available}.'
            ) from exc

    @staticmethod
    def available_datasets() -> list[str]:
        """
        List the DLML datasets that provide an input schema.

        Only these datasets can be validated against. The list comes from
        the installed version of DLML, so a schema added to the library
        becomes usable here without any change to BRAILS.

        Returns:
            list[str]:
                The identifiers of the datasets that provide a schema.
        """
        available = []

        for dataset_id in dlml.list_datasets():
            try:
                dlml.get_schema(dataset_id)
            except Exception:  # noqa: BLE001, S112
                continue
            available.append(dataset_id)

        return available

    def validate(self, inventory: AssetInventory) -> ValidationReport:
        """
        Check every asset, in every possible world, against the schema.

        A feature holding one value per possible world is checked one
        world at a time, so a violation that occurs in only some worlds is
        reported against those worlds alone. The inventory is not
        modified.

        Args:
            inventory (AssetInventory):
                The inventory to check.

        Returns:
            ValidationReport:
                The violations that were found.

        Raises:
            TypeError:
                If ``inventory`` is not an
                :class:`~brails.types.asset_inventory.AssetInventory`.
            BrailsError:
                If the inventory holds features whose number of
                realizations disagrees with its number of possible
                worlds.
        """
        if not InputValidator.is_inventory(inventory):
            raise TypeError(
                'Expected an instance of AssetInventory for inventory input.'
            )

        # A feature holding the wrong number of realizations cannot be
        # split into worlds, so reject it before reporting anything.
        inventory.validate_possible_worlds()

        n_pw = inventory.get_n_pw()
        report = ValidationReport(
            self.dataset_id, n_pw=n_pw, n_assets=len(inventory.inventory)
        )

        for world_index in range(n_pw):
            assets = {
                asset_id: _project_world(
                    features, world_index, inventory.vector_features
                )
                for asset_id, features in _iter_features(inventory)
            }
            for asset_id, messages in dlml.validate_assets(
                self.dataset_id, assets
            ).items():
                report.add(asset_id, world_index, messages)

        return report


def _iter_features(
    inventory: AssetInventory,
) -> list[tuple[str | int, dict[str, Any]]]:
    """
    Collect the features of every asset in an inventory.

    Args:
        inventory (AssetInventory):
            The inventory to read.

    Returns:
        list[tuple[str | int, dict[str, Any]]]:
            The identifier and features of each asset.
    """
    return [
        (asset_id, asset.features) for asset_id, asset in inventory.inventory.items()
    ]


def _project_world(
    features: dict[str, Any],
    world_index: int,
    vector_features: list[str],
) -> dict[str, Any]:
    """
    Reduce an asset's features to the values of one possible world.

    Features that hold a single value shared by every world are passed
    through unchanged.

    Args:
        features (dict[str, Any]):
            The features of one asset.
        world_index (int):
            The index of the possible world to extract.
        vector_features (list[str]):
            Names of the features whose values are vectors.

    Returns:
        dict[str, Any]:
            The feature values of the requested world.
    """
    return {
        key: (
            value[world_index]
            if is_probabilistic_feature(key, value, vector_features)
            else value
        )
        for key, value in features.items()
    }
