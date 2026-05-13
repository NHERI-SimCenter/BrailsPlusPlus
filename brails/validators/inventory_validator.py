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
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 08-18-2025
# 03-2026 fmk


"""
This module provides a utility class for validating AssetInventory objects.

.. autosummary::

      InventoryValidator
"""

from __future__ import annotations

from abc import ABC
from typing import Any
import pandas as pd


class InventoryValidator(ABC):
    """
    Base class for validating AssetInventory objects.

    This class can be subclassed to customize validation and repair logic for
    inventory-like inputs.
    """

    @staticmethod
    def is_inventory(inventory: Any) -> bool:
        """
        Check whether the given object is a valid AssetInventory.

        Args:
            inventory:
                The object to check.

        Returns:
            True if the object is an instance of AssetInventory, False otherwise.
        """
        # Lazy import to avoid circular import issues
        from brails.types.asset_inventory import AssetInventory
        return isinstance(inventory, AssetInventory)

    def fix_inventory(self, inventory: Any) -> tuple[Any, pd.DataFrame | None]:
        """
        Attempt to convert the input into a valid inventory.

        The default implementation does not modify the input. Subclasses may
        override this method to apply corrections and report issues.

        Args:
            inventory:
                The object to repair.

        Returns:
            A tuple containing:
                - fixed_inventory: the repaired or unchanged inventory
                - issues_df: a DataFrame describing any issues found or fixed,
                  or None if no issue report is produced
        """
        return inventory, None

    def validate_inventory(
        self, inventory: Any, fix_it: bool = False
    ) -> tuple[Any, pd.DataFrame | None]:
        """
        Validate that the input is a valid inventory.

        Args:
            inventory:
                The object to validate.
            fix_it:
                If True, attempt to repair invalid input before raising an error.

        Returns:
            A tuple containing:
                - valid_inventory: the validated or repaired inventory
                - issues_df: a DataFrame of issues/fixes, or None

        Raises:
            TypeError:
                If the input is invalid and cannot be repaired.
        """
        if fix_it:
            fixed_inventory, issues_df = self.fix_inventory(inventory)
            if self.is_inventory(fixed_inventory):
                return fixed_inventory, issues_df

        if self.is_inventory(inventory):
            return inventory, None

        raise TypeError(
            "Expected an instance of AssetInventory for inventory input."
        )
    
