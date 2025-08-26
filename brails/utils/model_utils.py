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
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 08-19-2025

"""
This module provides a utility class for computer vision models in BRAILS.

.. autosummary::

      ModelUtils
"""


from pathlib import Path
import torch


class ModelUtils:
    """
    Utility class for computer vision models in BRAILS.

    This class provides static methods to ensure necessary model files are
    available locally, downloading them if needed. Intended for use in
    applications that rely on pre-trained model weights.

    To use :class:`ModelUtils`, include this ``import`` statement in your code:

    .. code-block:: python

        from brails.utils import ModelUtils

    """

    @staticmethod
    def get_model_path(
        model_path: str = '',
        default_filename: str = '',
        download_url: str = '',
        model_description: str = "model",
        overwrite: bool = False,
    ) -> str:
        """
        Check if a model file is available locally, download it if necessary.

        Args:
            model_path (str, optional):
                Custom path to the model file. If provided, no download occurs.
            default_filename (str, optional):
                Filename to use if downloading the model.
            download_url (str, optional):
                URL to download the model if model does not exist locally.
            model_description (str, optional):
                Human-readable description of the model (default: ``'model'``).
            overwrite (bool, optional):
                If ``True``, re-download and overwrite the model file even if
                it exists. Defaults to ``False``.

        Returns:
            str: Absolute path to the model file.

        Raises:
            ValueError:
                If ``model_path`` is not provided and either
                ``default_filename`` or ``download_url`` is missing.

        Examples:
            Use a custom model path:

            >>> from brails.utils import ModelUtils
            >>> path = ModelUtils.get_model_path(
            ...     model_path='my_models/custom.pth'
            ... )
            Inferences will be performed using the custom model in
            my_models/custom.pth

            Download a default model if not already available:

            >>> path = ModelUtils.get_model_path(
            ...     default_filename='default_model.pth',
            ...     download_url=(
            ...         'https://zenodo.org/record/7271554/files/'
            ...         'trained_model_rooftype.pth'
            ...     ),
            ...     model_description='roof classification model'
            ... )
            Downloading default roof classification model to
            tmp/models/default_model.pth...
            100%|██████████| 77.9M/77.9M [04:11<00:00, 325kB/s]
            Default roof classification model successfully downloaded.

            Force overwrite an existing model file:

            >>> path = ModelUtils.get_model_path(
            ...     default_filename='default_model.pth',
            ...     download_url=(
            ...         'https://zenodo.org/record/7271554/files/'
            ...         'trained_model_rooftype.pth'
            ...     ),
            ...     model_description='roof classification model',
            ...     overwrite=True
            ... )
            Re-downloading default roof classification model to
            tmp/models/default_model.pth...
            100%|██████████| 77.9M/77.9M [04:12<00:00, 324kB/s]
            Default roof classification model successfully downloaded.
        """
        # Case 1 - User provides a custom model path:
        if model_path:
            print(
                f'Inferences will be performed using the custom '
                f'{model_description} in {model_path}'
            )
            return str(Path(model_path).resolve())

        # Case 2 - Use default download logic:
        if not default_filename or not download_url:
            raise ValueError(
                'If model_path is not provided, both default_filename and '
                'download_url must be specified.'
            )

        model_dir = Path("tmp/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        local_path = model_dir / default_filename

        if overwrite or not local_path.is_file():
            action = "Re-downloading" if overwrite and local_path.is_file() \
                else "Downloading"
            print(f'\n{action} default {model_description} to {local_path}...')
            torch.hub.download_url_to_file(
                download_url,
                str(local_path),
                progress=True
            )
            print(f'\nDefault {model_description} successfully downloaded.')
        else:
            print(
                f'Default {model_description} already available at '
                f'{local_path}. Inferences will be performed using this model.'
            )

        return str(local_path.resolve())
