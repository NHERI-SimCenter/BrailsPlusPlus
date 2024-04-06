import ast
import unittest
from unittest import mock
from brails import Importer
from brails import BrailsError


class TestImporter(unittest.TestCase):

    def test_package_not_found_raises_exception(self):
        with self.assertRaises(BrailsError):
            Importer('some_package_that_does_not_exist')

    def test_get_class_with_valid_name(self):
        importer = Importer('brails')
        cls = importer.get_class('AssetInventory')
        self.assertIsNotNone(cls)

    def test_get_class_with_invalid_name(self):
        importer = Importer('brails')
        with self.assertRaises(BrailsError) as context:
            importer.get_class(r'¯\_(ツ)_/¯')
        assert (
            f'is not found. These are the available classes: '
        ) in str(context.exception)

    @mock.patch('brails.Importer._find_package_path')
    def test_repr(self, mock_find_package_path):
        # Mock the method to return the path to the data directory
        mock_find_package_path.return_value = 'tests/utils/data/repr'

        importer = Importer('Brails')

        assert importer.__repr__() == (
            'Importer at tests/utils/data/repr\n'
            '2 available classes:\n'
            '  ClassA: brails.file_A\n'
            '  ClassB: brails.subdir.file_B\n'
            'Run help(<importer_object>) for usage info.'
        )

    @mock.patch('brails.Importer._find_package_path')
    def test_duplicate_classes_raises_exception(self, mock_find_package_path):
        # Mock the method to return the path to the data directory
        mock_find_package_path.return_value = 'tests/utils/data/duplicate_name'

        with self.assertRaises(BrailsError) as context:
            importer = Importer('Brails')

        self.assertIn('unique class names', str(context.exception))


if __name__ == '__main__':
    unittest.main()
