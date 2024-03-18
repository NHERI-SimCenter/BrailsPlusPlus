"""
Utility classes and methods for the brails module.

"""

from pathlib import Path
import importlib
import ast
import os


class BrailsError(Exception):
    """
    Custom exception for specific error cases in our application.
    """

    def __init__(self, message="An error occurred in the Brails application"):
        self.message = message
        super().__init__(self.message)


class Importer:
    """
    Dynamically parses the files of a specified package (without
    having to import anything) and provides access to the defined
    classes therein. Classes are imported on an as-needed basis.

    This class parses a given package directory, identifies all
    non-abstract classes, and makes them accessible through a dynamic
    import mechanism. It ensures that class names are unique within
    the package scope to avoid conflicts. Classes can be retrieved and
    instantiated by their name using the get_class method.

    Attributes
    ----------
    package_path (Path):
      The file system path to the root of the package.
    classes (dict):
      A dictionary mapping class names to their module paths.

    Raises
    ------
    BrailsError:
      If duplicate class names are found or if a specified class is
      not available.

    """

    def __init__(self, package_name='brails'):
        """
        Initialize the Importer, finding and parsing all classes in
        the 'brails' package.

        """
        self.package_path = self._find_package_path(package_name)
        self.classes = {}
        self._parse_package()

    def get_class(self, class_name):
        """
        Retrieve and import a class by its name.

        Parameters
        ----------
        class_name (str):
          The name of the class to retrieve.

        Returns
        -------
          The class object if found, otherwise raises BrailsError.

        Raises
        ------
        BrailsError:
          If the class cannot be found.

        """
        module_path = self.classes.get(class_name)
        if module_path:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        raise BrailsError(
            f'Class name `{class_name}` is not found. '
            f'These are the available classes: {self.classes}'
        )

    def _find_package_path(self, package_name):
        """
        Determines the file system path to the specified package.

        Parameters
        ----------
        package_name (str):
          The name of the package to locate.

        Returns
        -------
        Path:
          The path to the package.

        Raises
        ------
        BrailsError:
          If the package cannot be found.

        """
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
            return Path(spec.origin).parent
        raise BrailsError(f"Package {package_name} not found")

    def _parse_package(self):
        """
        Walk the package directory and parse each Python file to find
        classes.
        """
        for root, _, files in os.walk(self.package_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self._parse_file(file_path)

    def _parse_file(self, file_path):
        """
        Parse a single Python file to find class definitions.

        Parameters
        ----------
        file_path (str):
          The path to the file to parse.

        """
        relative_path = Path('brails') / os.path.relpath(
            file_path, self.package_path
        )
        module_path = os.path.splitext(relative_path)[0].replace(os.sep, '.')
        with open(file_path, 'r', encoding='utf-8') as file:
            node = ast.parse(file.read(), filename=file_path)
            for child in node.body:
                if isinstance(child, ast.ClassDef):
                    class_name = child.name
                    self._add_class(class_name, module_path)

    def _add_class(self, class_name, module_path):
        """
        Add a class to the internal dictionary, ensuring unique class
        names across modules.

        Parameters
        ----------
        class_name (str):
          The name of the class to add.
        module_path (str):
          The module path where the class is defined.

        Raises
        ------
        BrailsError:
          If the class name already exists in the dictionary.

        """
        # check that the class name does not already exist in some
        # other sub module
        if class_name in self.classes:
            raise BrailsError(
                f'Invalid module structure. '
                f'In Brails, we enforce a policy of unique class names. '
                f'Class name `{class_name}` is defined in both '
                f'`{self.classes[class_name]}` and `{module_path}`. '
                f'This is not allowed. '
                f'If you recently introduced a class, make sure '
                f'you specify a unique class name, no matter if '
                f'the module path is different. '
                f'Otherwise, please submit a bug report. '
            )
        # if all's good, add the class
        self.classes[class_name] = module_path

    def __repr__(self):
        """
        Return a string representation of the Importer, listing
        available classes and their modules.
        """
        class_list = '\n'.join(
            f"  {cls}: {mod}" for cls, mod in self.classes.items()
        )
        return (
            f"{self.__class__.__name__} at {self.package_path}"
            f"\n"
            f"{len(self.classes)} available classes:"
            f"\n"
            f"{class_list}"
            f"\n"
            f"Run help(<importer_object>) for usage info."
        )


# # DELETE THIS CODE
# # Usage example
# importer = Importer()
# my_class = importer.get_class('FacadeParser')
# instance = my_class({})
