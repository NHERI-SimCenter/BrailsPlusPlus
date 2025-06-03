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
# 06-03-2025

"""
This module provides a utility class for validating AssetInventory objects.

.. autosummary::

      InventoryValidator
"""
from brails.types.asset_inventory import AssetInventory


class InventoryValidator:
    """
    A utility class for validating AssetInventory objects.

    This class provides static methods for checking whether a given object is a
    valid  instance of `AssetInventory`, and for enforcing that validation with
    clear error reporting.

    Methods:
    is_inventory(inventory: AssetInventory) -> bool
        Returns True if the input is an instance of AssetInventory, otherwise
        False.

    validate_inventory(inventory: AssetInventory) -> None
        Raises a TypeError if the input is not an instance of AssetInventory.

    Examples:
    >>> from brails.utils.input_validator import InventoryValidator
    >>> inventory = AssetInventory(...)  # assume valid initialization
    >>> InventoryValidator.is_inventory(inventory)
    True
    >>> InventoryValidator.validate_inventory(inventory)
    # passes silently

    >>> InventoryValidator.validate_inventory("not an inventory")
    Traceback (most recent call last):
        ...
    TypeError: Expected an instance of AssetInventory for inventory input.
    """

    @staticmethod
    def is_inventory(inventory: AssetInventory) -> bool:
        """
        Check if the given object is an instance of AssetInventory.

        Args:
            inventory (AssetInventory):
                The object to check.

        Returns:
            bool:
                True if the object is an instance of AssetInventory, False
                otherwise.
        """
        return isinstance(inventory, AssetInventory)

    @staticmethod
    def validate_inventory(inventory: AssetInventory) -> None:
        """
        Validate that the input is an instance of AssetInventory.

        Args:
            inventory (AssetInventory):
                The object to validate.

        Raises:
            TypeError:
                If the input is not an instance of AssetInventory.
        """
        if not InventoryValidator.is_inventory(inventory):
            raise TypeError('Expected an instance of AssetInventory for '
                            'inventory input.')
