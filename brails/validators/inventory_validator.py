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
This module defines the base class for AssetInventory validators.

.. autosummary::

    InventoryValidator
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory
    from brails.validators.validation_report import ValidationReport


class InventoryValidator(ABC):
    """
    Base class for validating AssetInventory objects.

    A validator answers one question: does this inventory satisfy the
    input requirements of a simulation method? It reports what is
    wrong and changes nothing.

    Subclasses implement :meth:`validate`, which must not mutate the
    inventory it is given.

    To import the :class:`InventoryValidator` class, use:

    .. code-block:: python

        from brails.validators import InventoryValidator
    """

    @abstractmethod
    def validate(self, inventory: AssetInventory) -> ValidationReport:
        """
        Check an inventory against this validator's requirements.

        Implementations must treat ``inventory`` as read-only.

        Args:
            inventory (AssetInventory):
                The inventory to check.

        Returns:
            ValidationReport:
                The violations that were found. An empty report means the
                inventory satisfies the requirements.

        Raises:
            TypeError:
                If ``inventory`` is not an
                :class:`~brails.types.asset_inventory.AssetInventory`.
        """
