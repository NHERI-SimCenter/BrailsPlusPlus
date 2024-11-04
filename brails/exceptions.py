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
# 11-03-2024

"""This module defines the Brails exceptions."""


class BrailsError(Exception):
    """Custom exception for specific error cases in BRAILS."""

    def __init__(self,
                 message: str = 'An error occurred in the BRAILS application'):
        """
        Initialize a BrailsError with a specific message.

        Args:
            message (str): Description of the error.
        """
        self.message = message
        super().__init__(self.message)


class NotFoundError(BrailsError):
    """
    Exception raised when a required entity is not found.

    This exception should be used for missing elements like classes, keys, or
    parameters.

    Note: For missing files, use the built-in FileNotFoundError instead.
    """

    def __init__(self,
                 type_of_thing: str,
                 name: str,
                 where: str = None,
                 append: str = None):
        """
        Initialize a NotFoundError with context on the missing item.

        Args:
            type_of_thing (str): What the missing thing is, i.e.,
                class, key, parameter, etc.
            name (str): The name of the missing thing.
            where (str): Where the thing was expected to be found.
            append (str): Additional message to be appended.

        Example:
            >>> raise NotFoundError(
            ...     'class',
            ...     'ConvolutionFilter',
            ...     where='configuration_file'
            ... )
            NotFoundError: CLASS ConvolutionFilter is not found in
                configuration_file.

            Including additional information for context:
            >>> raise NotFoundError(
            ...     'class',
            ...     'ConvolutionFilter',
            ...     where='configuration_file',
            ...     append='Check config and retry.'
            ... )
            NotFoundError: CLASS ConvolutionFilter is not found in
                configuration_file. Check config and retry.

        """
        if where:
            add = f' in {where}'
        else:
            add = ''
        self.message = (f'{type_of_thing.capitalize()} {name} is not '
                        f'found{add}.')
        if append:
            self.message += '\n' + append
        super().__init__(self.message)
