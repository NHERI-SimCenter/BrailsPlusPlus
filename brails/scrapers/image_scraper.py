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
# Frank McKenna
# Barbaros Cetiner
#
# Last updated:
# 10-15-2024

# (subclass needed methods: _polygon_area and create_asset_inventory)

"""
This module defines the abstract ImageScraper class.

.. autosummary::

    ImageScraper
"""

from abc import ABC, abstractmethod

from brails.types.image_set import ImageSet
from brails.types.asset_inventory import AssetInventory


class ImageScraper(ABC):
    """
    Abstract base class for obtaining images for an AssetInventory.

      Method:
         get_images(inventory): An abstract method to return an ImageSet given
             an AssetInventory
    """

    @abstractmethod
    def get_images(self,
                   inventory: AssetInventory,
                   save_directory: str) -> ImageSet:
        """
        Abstract class that must be implemented by subclasses.

        This method will be used by the caller to obtain the images for assets
        in an area.

        Args:
            inventory (AssetInventory): AssetInventory for which the images
                will be retrieved.
            save_directory (str): Path to the folder where the retrieved images
                will be saved

        Returns:
            ImageSet: An ImageSet containing the images for the assets in the
                inventory.
        """
