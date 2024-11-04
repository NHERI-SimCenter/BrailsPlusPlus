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

"""
Utility classes and methods for the brails module.

.. autosummary::

      Importer
"""

from pathlib import Path
from typing import Any
import importlib.util
import ast
import os
from brails.exceptions import NotFoundError
from brails.exceptions import BrailsError


class Importer:
    """
    Dynamically parses files of a specified package and access its classes.

    This class parses a given package directory, identifies all
    non-abstract classes, and makes them accessible through a dynamic
    import mechanism. It ensures that class names are unique within
    the package scope to avoid conflicts. Classes can be retrieved and
    instantiated by their name using the get_class method.

    Attributes:
        package_path (Path): The file system path to the root of the package.
        max_parse_levels (int): Limits parsing of class files to the first
            `max_parse_levels` subdirectories.
        classes (dict): A dictionary mapping class names to their module paths.

    Raises:
        NotFoundError: If the specified package or class cannot be found.
        BrailsError: If duplicate class names are found in the code base.

    """

    def __init__(self, package_name="brails"):
        """
        Initialize Importer to find & parse all classes in specified package.

        Args:
            package_name (str) The name of the package to be parsed, by default
                "brails".
        """
        self.package_path = self._find_package_path(package_name)
        self.classes = {}
        self.max_parse_levels = 2
        self._parse_package()

    def get_object(self, json_object: dict[str, Any]) -> Any:
        """
        Create an instance of a class from JSON object data.

        Args:
            json_object (dict[str, Any]): A dictionary containing "classType"
                and "objData" keys.

        Returns:
            Any: An instance of the specified class, initialized with
                `objData`.

        Raises:
            NotFoundError: If "classType" or "objData" is missing, or if the
                class is not found.
        """
        class_type = json_object.get("classType")
        if class_type is None:
            raise NotFoundError(
                type_of_thing="key",
                name="`classType`",
                where="json data",
                append=f"Existing data: {json_object}",
            )

        python_class = self.get_class(class_type)
        if python_class is None:
            raise NotFoundError(
                type_of_thing="class of type",
                name=f"`{class_type}`",
                where="the framework",
            )

        object_data = json_object.get("objData")
        if object_data is None:
            raise NotFoundError(
                type_of_thing="key",
                name="`objData`",
                where="json data",
                append=f"Existing data: {json_object}",
            )

        return python_class(object_data)

    def get_class(self, class_name: str) -> Any:
        """
        Retrieve and import a class by its name.

        Args:
            class_name (str): The name of the class to retrieve.

        Returns:
            Any: The class object if found, otherwise raises BrailsError.

        Raises:
            NotFoundError: If the class cannot be found.

        """
        module_path = self.classes.get(class_name)
        if module_path:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)

        raise NotFoundError(
            type_of_thing="class",
            name=class_name,
            append=f"These are the available classes: {self.classes}",
        )

    def _find_package_path(self, package_name: str):
        """
        Determine the file system path to the specified package.

        Args:
            package_name (str): The name of the package to locate.

        Returns:
            Path: The path to the package.

        Raises:
            NotFoundError: If the package cannot be found.

        """
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
            return Path(spec.origin).parent
        raise NotFoundError(type_of_thing="package", name=package_name)

    def _parse_package(self):
        """Walk package directory & parse each Python file to find classes."""
        # root_directory = Path(self.package_path)
        for dirpath, _, files in os.walk(self.package_path):
            depth = dirpath.count(os.path.sep) \
                - str(self.package_path).count(os.path.sep)
            if depth <= self.max_parse_levels:
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(dirpath, file)
                        self._parse_file(file_path)

    def _parse_file(self, file_path: str):
        """
        Parse a single Python file to find class definitions.

        Args:
            file_path (str): The path to the file to parse.
        """
        relative_path = Path("brails") / \
            os.path.relpath(file_path, self.package_path)
        module_path = os.path.splitext(relative_path)[0].replace(os.sep, ".")
        with open(file_path, "r", encoding="utf-8") as file:
            node = ast.parse(file.read(), filename=file_path)
            for child in node.body:
                if isinstance(child, ast.ClassDef) and \
                        (not self._is_abstract(child)):
                    class_name = child.name
                    self._add_class(class_name, module_path)

    def _is_abstract(self, node: ast.ClassDef) -> bool:
        """
        Determine if a given AST class node represents an abstract class.

        Checks for inheritance from `abc.ABC` or usage of
        `abc.ABCMeta` as the metaclass, and the presence of any
        methods decorated with `@abstractmethod`.

        Args:
            node (ast.ClassDef): The AST node representing a class definition.

        Returns:
            bool: True if the class is abstract, False otherwise.

        """
        if isinstance(node, ast.ClassDef):
            # Check if it inherits from ABC or has ABCMeta as metaclass
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "ABC":
                    return True
                if isinstance(base, ast.Attribute) and base.attr == "ABCMeta":
                    return True

            # Check for any method with the @abstractmethod decorator
            for body_item in node.body:
                if isinstance(body_item,
                              (ast.FunctionDef, ast.AsyncFunctionDef)):

                    for decorator in body_item.decorator_list:
                        if (
                            isinstance(decorator, ast.Name)
                            and decorator.id == "abstractmethod"
                        ):
                            return True
        return False

    def _add_class(self, class_name: str, module_path: str):
        """
        Add a class to internal dictionary, ensuring unique module class names.

        Parameters:
            class_name (str): The name of the class to add.
            module_path (str): The module path where the class is defined.

        Raises:
            BrailsError: If the class name already exists in the dictionary.

        """
        # Check that the class name does not already exist in some
        # other sub module:
        if class_name in self.classes:
            raise BrailsError(
                f"Invalid module structure. "
                f"BRAILS requires each class name to be unique. "
                f"Class name `{class_name}` is defined in both "
                f"`{self.classes[class_name]}` and `{module_path}`. "
                f"This is not allowed. "
                f"If you recently introduced a class, make sure "
                f"you specify a unique class name, no matter if "
                f"the module path is different. "
                f"Otherwise, please submit a bug report. "
            )
        # If all is good, add the class:
        self.classes[class_name] = module_path

    def __repr__(self) -> str:
        """List available classes and their modules."""
        class_list = "\n".join(
            f"  {cls}: {mod}" for cls, mod in self.classes.items())
        return (
            f"{self.__class__.__name__} at {self.package_path}"
            f"\n"
            f"{len(self.classes)} available classes:"
            f"\n"
            f"{class_list}"
            f"\n"
            f"Run help(<importer_object>) for usage info."
        )
