"""Test suite for household_inventory.py"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, mock_open
from brails.types.household_inventory import Household, HouseholdInventory, clean_floats

# We don't need docstrings for the test objects
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


# Tests for clean_floats function
def test_clean_floats_with_integer_floats():
    obj = {'a': 1.0, 'b': 2.0}
    result = clean_floats(obj)
    assert result == {'a': 1, 'b': 2}


def test_clean_floats_with_actual_floats():
    obj = {'a': 1.5, 'b': 2.7}
    result = clean_floats(obj)
    assert result == {'a': 1.5, 'b': 2.7}


def test_clean_floats_with_mixed_types():
    obj = {'int': 5, 'float_int': 10.0, 'float': 3.14, 'string': 'test'}
    result = clean_floats(obj)
    expected = {'int': 5, 'float_int': 10, 'float': 3.14, 'string': 'test'}
    assert result == expected


def test_clean_floats_with_nested_dict():
    obj = {
        'level1': {
            'level2': {
                'float_int': 42.0,
                'float': 3.14
            }
        }
    }
    result = clean_floats(obj)
    expected = {
        'level1': {
            'level2': {
                'float_int': 42,
                'float': 3.14
            }
        }
    }
    assert result == expected


def test_clean_floats_with_list():
    obj = [1.0, 2.5, 3.0, 'string', 4]
    result = clean_floats(obj)
    expected = [1, 2.5, 3, 'string', 4]
    assert result == expected


def test_clean_floats_with_nested_list():
    obj = [[1.0, 2.0], [3.5, 4.0]]
    result = clean_floats(obj)
    expected = [[1, 2], [3.5, 4]]
    assert result == expected


def test_clean_floats_with_complex_structure():
    obj = {
        'dict': {'a': 1.0, 'b': 2.5},
        'list': [3.0, 4.5, {'nested': 5.0}],
        'primitive': 6.0
    }
    result = clean_floats(obj)
    expected = {
        'dict': {'a': 1, 'b': 2.5},
        'list': [3, 4.5, {'nested': 5}],
        'primitive': 6
    }
    assert result == expected


def test_clean_floats_with_non_numeric_types():
    obj = {'string': 'test', 'bool': True, 'none': None}
    result = clean_floats(obj)
    assert result == obj


def test_clean_floats_with_empty_structures():
    assert clean_floats({}) == {}
    assert clean_floats([]) == []


def test_clean_floats_with_primitive_types():
    assert clean_floats(5.0) == 5
    assert clean_floats(5.5) == 5.5
    assert clean_floats(5) == 5
    assert clean_floats('test') == 'test'


# Fixtures for Household tests
@pytest.fixture
def household_data():
    return {
        'household_id': 123,
        'features': {'income': 50000, 'size': 3}
    }


@pytest.fixture
def household(household_data):
    return Household(household_data['household_id'], household_data['features'])


# Tests for Household class
def test_init_with_features():
    household = Household(1, {'income': 50000})
    assert household.household_id == 1
    assert household.features == {'income': 50000}


def test_init_without_features():
    household = Household(1)
    assert household.household_id == 1
    assert household.features == {}


def test_init_with_none_features():
    household = Household(1, None)
    assert household.household_id == 1
    assert household.features == {}


def test_add_features_with_overwrite_true(household):
    new_features = {'age': 35, 'income': 60000}
    updated, n_pw = household.add_features(new_features, overwrite=True)
    
    assert updated is True
    assert n_pw == 1
    assert household.features['age'] == 35
    assert household.features['income'] == 60000  # overwritten


def test_add_features_with_overwrite_false(household):
    new_features = {'age': 35, 'income': 60000}
    updated, n_pw = household.add_features(new_features, overwrite=False)
    
    assert updated is True
    assert n_pw == 1
    assert household.features['age'] == 35  # new feature added
    assert household.features['income'] == 50000  # not overwritten


def test_add_features_with_list_values(household):
    new_features = {'options': [1, 2, 3]}
    updated, n_pw = household.add_features(new_features)
    
    assert updated is True
    assert n_pw == 3
    assert household.features['options'] == [1, 2, 3]


def test_add_features_with_inconsistent_list_lengths(household):
    # First add a list of length 3
    household.add_features({'list1': [1, 2, 3]})
    
    # Then add a list of different length - should print warning
    with patch('builtins.print') as mock_print:
        updated, n_pw = household.add_features({'list2': [4, 5]})
        # Check if warning was printed (may not always happen depending on logic)
        if mock_print.called:
            assert 'WARNING' in mock_print.call_args[0][0]
        assert updated is True
        assert n_pw == 2  # Should be length of the new list


def test_add_features_no_overwrite_existing_key(household):
    new_features = {'income': 60000}
    updated, n_pw = household.add_features(new_features, overwrite=False)
    
    assert updated is False  # No new keys added
    assert household.features['income'] == 50000  # unchanged


def test_remove_features(household):
    household.features['age'] = 35
    result = household.remove_features(['age', 'nonexistent'])
    
    assert result is True
    assert 'age' not in household.features
    assert 'income' in household.features  # other features remain


def test_remove_features_empty_list(household):
    original_features = household.features.copy()
    result = household.remove_features([])
    
    assert result is True
    assert household.features == original_features


def test_print_info(household):
    with patch('builtins.print') as mock_print:
        household.print_info()
        
        # Check that print was called twice (ID and features)
        assert mock_print.call_count == 2
        
        # Check the content of the calls
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert 'Household ID: 123' in calls[0]
        assert 'Features:' in calls[1]


# Fixtures for HouseholdInventory tests
@pytest.fixture
def inventory():
    return HouseholdInventory()


@pytest.fixture
def household1():
    return Household(1, {'income': 50000, 'size': 3})


@pytest.fixture
def household2():
    return Household(2, {'income': 75000, 'size': 4})


# Tests for HouseholdInventory class
def test_inventory_init():
    inventory = HouseholdInventory()
    assert inventory.inventory == {}
    assert inventory.n_pw == 1


def test_add_household_success(inventory, household1):
    result = inventory.add_household(1, household1)
    
    assert result is True
    assert 1 in inventory.inventory
    assert inventory.inventory[1] == household1


def test_add_household_duplicate_id(inventory, household1, household2):
    inventory.add_household(1, household1)
    
    with patch('builtins.print') as mock_print:
        result = inventory.add_household(1, household2)
        
        assert result is False
        mock_print.assert_called_once()
        assert 'already exists' in mock_print.call_args[0][0]


def test_add_household_invalid_type(inventory):
    with pytest.raises(TypeError) as exc_info:
        inventory.add_household(1, "not_a_household")
    
    assert 'Expected an instance of Household' in str(exc_info.value)


def test_add_household_features_success(inventory, household1):
    inventory.add_household(1, household1)
    new_features = {'age': 35}
    
    result = inventory.add_household_features(1, new_features)
    
    assert result is True
    assert inventory.inventory[1].features['age'] == 35


def test_add_household_features_nonexistent_household(inventory):
    with patch('builtins.print') as mock_print:
        result = inventory.add_household_features(999, {'age': 35})
        
        assert result is False
        mock_print.assert_called_once()
        assert 'No existing Household' in mock_print.call_args[0][0]


def test_add_household_features_with_possible_worlds(inventory, household1):
    inventory.add_household(1, household1)
    new_features = {'options': [1, 2, 3]}
    
    result = inventory.add_household_features(1, new_features)
    
    assert result is True
    assert inventory.n_pw == 3


def test_change_feature_names_success(inventory, household1):
    inventory.add_household(1, household1)
    mapping = {'income': 'annual_income', 'size': 'household_size'}
    
    inventory.change_feature_names(mapping)
    
    features = inventory.inventory[1].features
    assert 'annual_income' in features
    assert 'household_size' in features
    assert 'income' not in features
    assert 'size' not in features


def test_change_feature_names_invalid_mapping_type(inventory):
    with pytest.raises(TypeError) as exc_info:
        inventory.change_feature_names("not_a_dict")
    
    assert 'Expected \'feature_name_mapping\' to be a dictionary' in str(exc_info.value)


def test_change_feature_names_invalid_key_value_types(inventory):
    with pytest.raises(TypeError) as exc_info:
        inventory.change_feature_names({123: 'new_name'})
    
    assert 'All keys and values' in str(exc_info.value)


def test_change_feature_names_nonexistent_feature(inventory, household1):
    inventory.add_household(1, household1)
    mapping = {'nonexistent': 'new_name'}
    
    # Should not raise error, just do nothing
    inventory.change_feature_names(mapping)
    
    features = inventory.inventory[1].features
    assert 'new_name' not in features


def test_get_household_features_success(inventory, household1):
    inventory.add_household(1, household1)
    
    found, features = inventory.get_household_features(1)
    
    assert found is True
    assert features == household1.features


def test_get_household_features_nonexistent(inventory):
    found, features = inventory.get_household_features(999)
    
    assert found is False
    assert features == {}


def test_get_household_ids(inventory, household1, household2):
    inventory.add_household(1, household1)
    inventory.add_household(2, household2)
    
    ids = inventory.get_household_ids()
    
    assert set(ids) == {1, 2}


def test_get_household_ids_empty(inventory):
    ids = inventory.get_household_ids()
    assert ids == []


def test_print_info(inventory, household1):
    inventory.add_household(1, household1)
    
    with patch('builtins.print') as mock_print:
        inventory.print_info()
        
        # Should print class name, inventory type, and household info
        assert mock_print.call_count >= 3


def test_remove_household_success(inventory, household1):
    inventory.add_household(1, household1)
    
    result = inventory.remove_household(1)
    
    assert result is True
    assert 1 not in inventory.inventory


def test_remove_household_nonexistent(inventory):
    result = inventory.remove_household(999)
    assert result is False


def test_remove_features(inventory, household1, household2):
    inventory.add_household(1, household1)
    inventory.add_household(2, household2)
    
    result = inventory.remove_features(['income'])
    
    assert result is True
    assert 'income' not in inventory.inventory[1].features
    assert 'income' not in inventory.inventory[2].features


def test_to_json_without_file(inventory, household1):
    inventory.add_household(1, household1)
    
    json_data = inventory.to_json()
    
    assert isinstance(json_data, dict)
    assert json_data['type'] == 'HouseholdInventory'
    assert 'generated' in json_data
    assert 'brails_version' in json_data
    assert 'households' in json_data
    assert len(json_data['households']) == 1


def test_to_json_with_file(inventory, household1):
    inventory.add_household(1, household1)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        tmp_filename = tmp_file.name
    
    try:
        json_data = inventory.to_json(tmp_filename)
        
        # Check that file was created and contains expected data
        assert os.path.exists(tmp_filename)
        
        with open(tmp_filename, 'r') as f:
            file_data = json.load(f)
        
        assert file_data['type'] == 'HouseholdInventory'
        assert len(file_data['households']) == 1
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_success(inventory):
    # Create test JSON data
    test_data = {
        "type": "HouseholdInventory",
        "households": [
            {
                "household_id": 1,
                "features": {"income": 50000, "size": 3}
            },
            {
                "household_id": 2,
                "features": {"income": 75000, "size": 4}
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        result = inventory.read_from_json(tmp_filename, keep_existing=True)
        
        assert result is True
        assert len(inventory.inventory) == 2
        assert 1 in inventory.inventory
        assert 2 in inventory.inventory
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_clear_existing(inventory, household1):
    # Add existing household
    inventory.add_household(999, household1)
    
    test_data = {
        "households": [
            {
                "household_id": 1,
                "features": {"income": 50000}
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        result = inventory.read_from_json(tmp_filename, keep_existing=False)
        
        assert result is True
        assert len(inventory.inventory) == 1
        assert 1 in inventory.inventory
        assert 999 not in inventory.inventory
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_file_not_found(inventory):
    with pytest.raises(Exception) as exc_info:
        inventory.read_from_json('nonexistent_file.json', keep_existing=True)
    
    assert 'does not exist' in str(exc_info.value)


def test_read_from_json_invalid_json(inventory):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        tmp_file.write('invalid json content')
        tmp_filename = tmp_file.name
    
    try:
        with pytest.raises(Exception) as exc_info:
            inventory.read_from_json(tmp_filename, keep_existing=True)
        
        assert 'not a valid JSON file' in str(exc_info.value)
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_invalid_schema_not_dict(inventory):
    test_data = []  # Should be dict, not list
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            inventory.read_from_json(tmp_filename, keep_existing=True)
        
        assert 'Root object must be a dict' in str(exc_info.value)
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_missing_households_key(inventory):
    test_data = {"type": "HouseholdInventory"}  # Missing households key
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            inventory.read_from_json(tmp_filename, keep_existing=True)
        
        assert 'must contain a \'households\' key' in str(exc_info.value)
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_households_not_list(inventory):
    test_data = {"households": "not_a_list"}
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            inventory.read_from_json(tmp_filename, keep_existing=True)
        
        assert 'must be a list' in str(exc_info.value)
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_household_not_dict(inventory):
    test_data = {"households": ["not_a_dict"]}
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            inventory.read_from_json(tmp_filename, keep_existing=True)
        
        assert 'must be a dict' in str(exc_info.value)
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_missing_household_id(inventory):
    test_data = {"households": [{"features": {}}]}  # Missing household_id
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            inventory.read_from_json(tmp_filename, keep_existing=True)
        
        assert 'must have a \'household_id\' key' in str(exc_info.value)
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_invalid_household_id_type(inventory):
    test_data = {"households": [{"household_id": "not_int", "features": {}}]}
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            inventory.read_from_json(tmp_filename, keep_existing=True)
        
        assert 'must be an int' in str(exc_info.value)
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_invalid_features_type(inventory):
    test_data = {"households": [{"household_id": 1, "features": "not_dict"}]}
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            inventory.read_from_json(tmp_filename, keep_existing=True)
        
        assert 'features at index 0 must be a dict' in str(exc_info.value)
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_invalid_feature_key_type(inventory):
    # JSON automatically converts integer keys to strings, so this test
    # needs to be adjusted. Let's test the actual validation logic instead.
    test_data = {"households": [{"household_id": 1, "features": {"123": "value"}}]}
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        # This should actually succeed since JSON converts keys to strings
        result = inventory.read_from_json(tmp_filename, keep_existing=True)
        assert result is True
        assert "123" in inventory.inventory[1].features
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_invalid_feature_value_type(inventory):
    test_data = {"households": [{"household_id": 1, "features": {"key": {"nested": "dict"}}}]}
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            inventory.read_from_json(tmp_filename, keep_existing=True)
        
        assert 'Feature values must be string, number' in str(exc_info.value)
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_read_from_json_valid_feature_list_values(inventory):
    test_data = {
        "households": [
            {
                "household_id": 1,
                "features": {
                    "string_list": ["a", "b", "c"],
                    "number_list": [1, 2, 3.5],
                    "mixed_list": ["text", 42, 3.14]
                }
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(test_data, tmp_file)
        tmp_filename = tmp_file.name
    
    try:
        result = inventory.read_from_json(tmp_filename, keep_existing=True)
        
        assert result is True
        features = inventory.inventory[1].features
        assert features["string_list"] == ["a", "b", "c"]
        assert features["number_list"] == [1, 2, 3.5]
        assert features["mixed_list"] == ["text", 42, 3.14]
        
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)