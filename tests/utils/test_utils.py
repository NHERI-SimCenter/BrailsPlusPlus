import unittest
from unittest import mock
from brails import Importer
from brails.exceptions import BrailsError
from brails.exceptions import NotFoundError


class TestImporter(unittest.TestCase):

    def setUp(self):
        self.importer = Importer('brails')

    def test_package_not_found_raises_exception(self):
        with self.assertRaises(BrailsError):
            Importer('some_package_that_does_not_exist')

    def test_get_class_with_valid_name(self):
        cls = self.importer.get_class('AssetInventory')
        self.assertIsNotNone(cls)

    def test_get_class_with_invalid_name(self):
        with self.assertRaises(BrailsError) as context:
            self.importer.get_class(r'¯\_(ツ)_/¯')
        assert ('is not found.\nThese are the available classes: ') in str(
            context.exception
        )

    def test_get_object_no_class_type_raises_error(self):
        json_object = {}
        with self.assertRaises(NotFoundError) as context:
            self.importer.get_object(json_object)
        self.assertIn('`classType`', str(context.exception))

        json_object = {'classType': None}
        with self.assertRaises(NotFoundError) as context:
            self.importer.get_object(json_object)
        self.assertIn('`classType`', str(context.exception))

    def test_get_object_invalid_class_type_raises_error(self):
        json_object = {'classType': 'NonExistentClass'}
        with self.assertRaises(NotFoundError) as context:
            self.importer.get_object(json_object)
        self.assertIn('NonExistentClass', str(context.exception))

    def test_get_object_no_obj_data_raises_error(self):
        # `Importer` is a reliable "valid" class
        # but `objData` is missing
        json_object = {'classType': 'Importer'}
        with self.assertRaises(NotFoundError) as context:
            self.importer.get_object(json_object)
        self.assertIn('`objData`', str(context.exception))

    # Important: please note that we need to flip the order of
    # the mock variables in the method declaration, compared with
    # the order of the decorators.
    @mock.patch('brails.Importer._find_package_path')
    @mock.patch('brails.Importer.get_class')
    def test_get_object_success(self, mock_get_class, mock_find_package_path):
        # Mock the method to return the path to the data directory
        mock_find_package_path.return_value = 'tests/utils/data/success'
        module = __import__(
            'tests.utils.data.success.file_A',
            fromlist=['ClassA'],
        )
        mock_get_class.return_value = getattr(module, 'ClassA')
        self.importer = Importer('brails')  # replace previous Importer
        json_object = {'classType': 'ClassA', 'objData': 'test_data'}
        result = self.importer.get_object(json_object)
        self.assertEqual(result.some_data, 'test_data')

    @mock.patch('brails.Importer._find_package_path')
    def test_repr(self, mock_find_package_path):
        # Mock the method to return the path to the data directory
        mock_find_package_path.return_value = 'tests/utils/data/repr'
        self.importer = Importer('brails')  # replace previous Importer

        assert self.importer.__repr__() == (
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
            Importer('Brails')
        self.assertIn('unique class names', str(context.exception))


if __name__ == '__main__':

    # Debugging --- test a specific method
    suite = unittest.TestSuite()
    suite.addTest(
        TestImporter('test_get_object_invalid_class_type_raises_error')
    )
    runner = unittest.TextTestRunner()
    runner.run(suite)

    unittest.main()
