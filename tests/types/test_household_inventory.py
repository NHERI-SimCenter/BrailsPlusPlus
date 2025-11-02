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
# Adam Zsarnoczay

"""Tests for Household and HouseholdInventory in brails.types.household_inventory.

These tests cover class CRUD operations, I/O/serialization, and core logic
such as ID generation and inventory merging. Fixtures provide reusable
Household and HouseholdInventory instances.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError as _PkgNotFound  # type: ignore
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Any, Dict, List
from unittest.mock import patch

import pytest

from brails.types.household_inventory import Household, HouseholdInventory

if TYPE_CHECKING:  # pragma: no cover - used only for typing annotations
    from pathlib import Path

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def sample_features() -> Dict[str, Any]:
    """Provide a sample features dictionary used in multiple tests."""
    return {'IncomeSample': 50000, 'NumberOfPersons': 3, 'Family': False}


@pytest.fixture
def empty_household() -> Household:
    """Provide an empty Household instance for tests."""
    return Household()


@pytest.fixture
def populated_household(sample_features: Dict[str, Any]) -> Household:
    """Provide a Household pre-populated with sample features."""
    return Household(features=sample_features.copy())


# -----------------------------
# 1. Household Class Tests
# -----------------------------


# test_household_initialization
@pytest.mark.parametrize(
    'features',
    [
        {'a': 1},
        {},
    ],
)
def test_household_initialization_valid(features: Dict[str, Any]) -> None:
    """Test that valid feature dictionaries initialize Household correctly."""
    # Arrange & Act
    hh = Household(features=features)

    # Assert
    assert isinstance(hh, Household)
    assert hh.features == features


def test_household_initialization_default_empty() -> None:
    """Test that default constructor creates an empty features dict."""
    # Act
    hh = Household()

    # Assert
    assert isinstance(hh, Household)
    assert hh.features == {}


@pytest.mark.parametrize(
    'bad_features',
    [
        [1, 2, 3],
        'not a dict',
        123,
        1.23,
        {1, 2},
        ('a', 'b'),
        object(),
    ],
)
def test_household_initialization_type_error(bad_features: Any) -> None:
    """Test that non-dict features raise TypeError in Household.__init__."""
    # Act / Assert
    with pytest.raises(TypeError):
        Household(features=bad_features)  # type: ignore[arg-type]


# test_household_add_features
@pytest.mark.parametrize(
    'bad_input',
    [
        [1, 2],
        'bad',
        1,
        1.2,
        None,
        {1, 2},
        ('a',),
        object(),
    ],
)
def test_add_features_type_error(
    populated_household: Household, bad_input: Any
) -> None:
    """Test add_features raises TypeError for non-dict inputs."""
    with pytest.raises(TypeError):
        populated_household.add_features(bad_input)  # type: ignore[arg-type]


def test_add_features_overwrite_full() -> None:
    """Test overwrite=True updates all existing keys."""
    # Arrange
    hh = Household(features={'a': 1, 'b': 2})
    additional = {'a': 10, 'b': 20}

    # Act
    updated = hh.add_features(additional, overwrite=True)

    # Assert
    assert updated is True
    assert hh.features == {'a': 10, 'b': 20}


def test_add_features_overwrite_partial() -> None:
    """Test overwrite=True updates existing keys and adds new ones."""
    hh = Household(features={'a': 1})
    updated = hh.add_features({'a': 2, 'b': 3}, overwrite=True)
    assert updated is True
    assert hh.features == {'a': 2, 'b': 3}


def test_add_features_overwrite_new_only() -> None:
    """Test overwrite=True adds all-new features to an empty household."""
    hh = Household(features={})
    updated = hh.add_features({'x': 1}, overwrite=True)
    assert updated is True
    assert hh.features == {'x': 1}


def test_add_features_no_overwrite_preserves_existing_and_adds_new() -> None:
    """Test overwrite=False preserves existing values and adds new keys."""
    # Arrange
    hh = Household(features={'a': 1})

    # Act
    updated = hh.add_features({'a': 2, 'b': 3}, overwrite=False)

    # Assert
    assert updated is True  # new key added
    assert hh.features == {'a': 1, 'b': 3}


def test_add_features_no_overwrite_no_changes_returns_false() -> None:
    """Test overwrite=False returns False when no new keys are added."""
    hh = Household(features={'a': 1})
    updated = hh.add_features({'a': 2}, overwrite=False)
    assert updated is False
    assert hh.features == {'a': 1}


# test_household_remove_features
@pytest.mark.parametrize(
    ('to_remove', 'expected'),
    [
        (['IncomeSample'], {'NumberOfPersons': 3, 'Family': False}),
        (['IncomeSample', 'NumberOfPersons'], {'Family': False}),
        (
            ['missing'],
            {'IncomeSample': 50000, 'NumberOfPersons': 3, 'Family': False},
        ),  # silent if missing
    ],
)
def test_remove_features_removes_as_expected(
    sample_features: Dict[str, Any], to_remove: List[str], expected: Dict[str, Any]
) -> None:
    """Test Household.remove_features removes keys and ignores missing ones."""
    hh = Household(features=sample_features)
    hh.remove_features(to_remove)
    assert hh.features == expected


@pytest.mark.parametrize(
    'bad_list',
    [
        'not a list',
        123,
        1.23,
        None,
        [1, 2, 3],
        ['a', 1],
        {'a'},
        ('a',),
        object(),
    ],
)
def test_remove_features_type_error(
    populated_household: Household, bad_list: Any
) -> None:
    """Test remove_features raises TypeError for non-list-of-str inputs."""
    with pytest.raises(TypeError):
        populated_household.remove_features(bad_list)  # type: ignore[arg-type]


# test_household_print_info


def test_print_info_outputs_json() -> None:
    """Test that print_info prints the features JSON string once."""
    # Arrange
    features = {'alpha': 1, 'beta': True}
    hh = Household(features=features)
    expected_str = f'\t Features: {json.dumps(features, indent=2)}'

    with patch('builtins.print') as mocked_print:
        # Act
        hh.print_info()

        # Assert
        mocked_print.assert_called_once_with(expected_str)


# -----------------------------
# 2. HouseholdInventory Class Tests
# -----------------------------

# Fixtures for inventories


@pytest.fixture
def empty_inventory() -> HouseholdInventory:
    """Provide an empty HouseholdInventory instance for tests."""
    return HouseholdInventory()


@pytest.fixture
def populated_inventory(sample_features: Dict[str, Any]) -> HouseholdInventory:
    """Provide a HouseholdInventory with two predefined households."""
    inv = HouseholdInventory()
    inv.add_household('H1', Household(features={'a': 1, 'b': 2}))
    inv.add_household('H2', Household(features=sample_features.copy()))
    return inv


# test_inventory_initialization


def test_inventory_initialization(empty_inventory: HouseholdInventory) -> None:
    """Test that a new inventory starts with an empty dictionary."""
    assert isinstance(empty_inventory, HouseholdInventory)
    assert empty_inventory.inventory == {}


# test_inventory_add_household


def test_inventory_add_valid_household(empty_inventory: HouseholdInventory) -> None:
    """Test adding a valid Household stores it under the given ID."""
    hh = Household(features={'x': 1})
    empty_inventory.add_household('ID1', hh)
    assert empty_inventory.inventory['ID1'] is hh


@pytest.mark.parametrize('bad_obj', [None, 123, 'hh', {'features': {}}, object()])
def test_inventory_add_household_type_error(
    empty_inventory: HouseholdInventory, bad_obj: Any
) -> None:
    """Test add_household raises TypeError when given a non-Household object."""
    with pytest.raises(TypeError):
        empty_inventory.add_household('ID2', bad_obj)  # type: ignore[arg-type]


def test_inventory_add_household_overwrite_false_ignores_duplicate(
    empty_inventory: HouseholdInventory,
) -> None:
    """Test overwrite=False ignores duplicate household IDs and keeps original."""
    hh1 = Household(features={'v': 1})
    hh2 = Household(features={'v': 2})
    empty_inventory.add_household('ID', hh1, overwrite=False)
    empty_inventory.add_household('ID', hh2, overwrite=False)
    # Expect original remains
    assert empty_inventory.inventory['ID'].features == {'v': 1}


def test_inventory_add_household_overwrite_true_replaces_duplicate(
    empty_inventory: HouseholdInventory,
) -> None:
    """Test overwrite=True replaces existing household with the new one."""
    hh1 = Household(features={'v': 1})
    hh2 = Household(features={'v': 2})
    empty_inventory.add_household('ID', hh1, overwrite=False)
    empty_inventory.add_household('ID', hh2, overwrite=True)
    # Expect replacement
    assert empty_inventory.inventory['ID'].features == {'v': 2}


# test_inventory_change_feature_names


def test_inventory_change_feature_names_basic(
    populated_inventory: HouseholdInventory,
) -> None:
    """Test renaming features across households updates keys as expected."""
    # Rename 'a' to 'alpha' in H1 and 'IncomeSample' to 'AnnualIncome' in H2
    populated_inventory.change_feature_names(
        {'a': 'alpha', 'IncomeSample': 'AnnualIncome'}
    )
    assert 'alpha' in populated_inventory.inventory['H1'].features
    assert 'a' not in populated_inventory.inventory['H1'].features
    assert 'AnnualIncome' in populated_inventory.inventory['H2'].features
    assert 'IncomeSample' not in populated_inventory.inventory['H2'].features


def test_inventory_change_feature_names_skips_nonexistent(
    populated_inventory: HouseholdInventory,
) -> None:
    """Test renaming skips missing features without raising errors."""
    # 'does_not_exist' should be ignored
    populated_inventory.change_feature_names({'does_not_exist': 'new_name'})
    # Ensure nothing crashed and original features remain
    assert 'does_not_exist' not in populated_inventory.inventory['H1'].features
    assert 'new_name' not in populated_inventory.inventory['H1'].features


@pytest.mark.parametrize(
    'bad_mapping',
    [
        None,
        123,
        'bad',
        [('a', 'b')],
        {'a': 1},
        {1: 'b'},
        {'a': 'b', 2: 'c'},
    ],
)
def test_inventory_change_feature_names_type_error(
    populated_inventory: HouseholdInventory, bad_mapping: Any
) -> None:
    """Test TypeError is raised for invalid mapping inputs or key/value types."""
    with pytest.raises(TypeError):
        populated_inventory.change_feature_names(bad_mapping)  # type: ignore[arg-type]


def test_inventory_change_feature_names_raises_name_error_on_conflict(
    populated_inventory: HouseholdInventory,
) -> None:
    """Test renaming to an existing feature raises NameError (conflict)."""
    # Create conflict by renaming an existing key to an already existing one in the same household
    # In H1 we have features {"a":1, "b":2}; renaming 'a' -> 'b' should raise NameError
    with pytest.raises(NameError):
        populated_inventory.change_feature_names({'a': 'b'})


# test_inventory_remove_features


def test_inventory_remove_features_across_all_households(
    populated_inventory: HouseholdInventory,
) -> None:
    """Test removing features across all households works as intended."""
    # Remove from both H1 and H2 where applicable
    populated_inventory.remove_features(
        ['b', 'NumberOfPersons']
    )  # 'b' in H1; 'NumberOfPersons' in H2
    assert 'b' not in populated_inventory.inventory['H1'].features
    assert 'NumberOfPersons' not in populated_inventory.inventory['H2'].features


@pytest.mark.parametrize(
    'bad_list',
    [
        'not a list',
        123,
        1.23,
        None,
        [1, 2, 3],
        ['a', 1],
        {'a'},
        ('a',),
        object(),
    ],
)
def test_inventory_remove_features_type_error(
    populated_inventory: HouseholdInventory, bad_list: Any
) -> None:
    """Test inventory.remove_features raises TypeError for invalid lists."""
    with pytest.raises(TypeError):
        populated_inventory.remove_features(bad_list)  # type: ignore[arg-type]


# test_inventory_get_household_ids


def test_inventory_get_household_ids_empty(
    empty_inventory: HouseholdInventory,
) -> None:
    """Test get_household_ids returns an empty list for empty inventory."""
    assert empty_inventory.get_household_ids() == []


def test_inventory_get_household_ids_populated(
    populated_inventory: HouseholdInventory,
) -> None:
    """Test get_household_ids returns all current household IDs."""
    ids = populated_inventory.get_household_ids()
    assert set(ids) == {'H1', 'H2'}


# test_inventory_remove_household


def test_inventory_remove_household_existing(
    populated_inventory: HouseholdInventory,
) -> None:
    """Test removing an existing household deletes it from the inventory."""
    populated_inventory.remove_household('H1')
    assert 'H1' not in populated_inventory.inventory


def test_inventory_remove_household_nonexistent(
    populated_inventory: HouseholdInventory,
) -> None:
    """Test removing a missing household is a no-op and does not raise."""
    # Should not raise and should not change existing households
    populated_inventory.remove_household('H999')
    assert set(populated_inventory.inventory.keys()) == {'H1', 'H2'}


# test_inventory_print_info


def test_inventory_print_info_outputs(
    populated_inventory: HouseholdInventory,
) -> None:
    """Test that print_info prints summary lines and each household key."""
    with patch('builtins.print') as mocked_print:
        populated_inventory.print_info()

    # Verify key parts of the output were printed
    printed_lines = [
        ' '.join(str(x) for x in args) for args, _ in mocked_print.call_args_list
    ]
    assert HouseholdInventory.__name__ in printed_lines[0]
    assert any('Inventory stored in' in s for s in printed_lines)
    assert any('Key:  H1 Household:' in s for s in printed_lines)
    assert any('Key:  H2 Household:' in s for s in printed_lines)


# -----------------------------
# 3. I/O and Serialization Tests
# -----------------------------


def _compare_inventories(inv1: HouseholdInventory, inv2: HouseholdInventory) -> None:
    """Compare two inventories for equality of keys and feature dictionaries."""
    assert set(inv1.inventory.keys()) == set(inv2.inventory.keys())
    for k in inv1.inventory:
        assert inv1.inventory[k].features == inv2.inventory[k].features


# test_inventory_to_json


def test_inventory_to_json_empty_no_file_write(
    empty_inventory: HouseholdInventory, tmp_path: Path
) -> None:
    """Test to_json on empty inventory and that no file is written by default."""
    # Act
    data = empty_inventory.to_json()

    # Assert JSON structure
    assert data['type'] == 'HouseholdInventory'
    assert 'generated' in data
    assert isinstance(data['generated'], str)
    assert 'brails_version' in data
    assert isinstance(data['brails_version'], str)
    assert data['households'] == {}

    # Ensure no file is created when no output_file is provided
    out_file = tmp_path / 'should_not_exist.json'
    assert not out_file.exists()


def test_inventory_to_json_populated_and_file_write(tmp_path: Path) -> None:
    """Test to_json writes a file and contains the expected JSON structure."""
    # Arrange: inventory with string and int keys
    inv = HouseholdInventory()
    inv.add_household('A', Household(features={'x': 1}))
    inv.add_household(1, Household(features={'y': 2}))

    # Act: write to file
    out_path = tmp_path / 'inventory.json'
    data = inv.to_json(str(out_path))

    # Assert returned data structure
    assert set(data.keys()) >= {'type', 'generated', 'brails_version', 'households'}
    # Keys must be strings in JSON
    assert set(data['households'].keys()) == {'A', '1'}
    assert data['households']['A'] == {'x': 1}
    assert data['households']['1'] == {'y': 2}

    # File exists and contents parse to same structure
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding='utf-8'))
    assert on_disk['households'] == data['households']


def test_inventory_to_json_sets_brails_version_NA_when_package_missing() -> None:  # noqa: N802
    """Test that missing BRAILS package sets brails_version to 'NA'."""
    inv = HouseholdInventory()

    # Patch the function where it is used in the module
    with patch('brails.types.household_inventory.version', side_effect=_PkgNotFound):
        data = inv.to_json()

    assert data['brails_version'] == 'NA'


# test_inventory_read_from_json


def test_inventory_read_from_json_from_dict_and_string_and_file(
    tmp_path: Path,
) -> None:
    """Test read_from_json from dict, JSON string, and file path variants."""
    # Arrange valid dict matching schema
    valid_dict = {
        'type': 'HouseholdInventory',
        'generated': '2025-01-01T00:00:00Z',
        'brails_version': '1.0',
        'households': {'H1': {'a': 1, 'b': 2}, '2': {'IncomeSample': 50000}},
    }

    # Load from dict
    inv1 = HouseholdInventory()
    inv1.read_from_json(valid_dict)

    # Load from JSON string
    inv2 = HouseholdInventory()
    inv2.read_from_json(json.dumps(valid_dict))

    # Write file then load from file path
    json_path = tmp_path / 'inv.json'
    json_path.write_text(json.dumps(valid_dict), encoding='utf-8')
    inv3 = HouseholdInventory()
    inv3.read_from_json(str(json_path))

    # All three should be equivalent
    _compare_inventories(inv1, inv2)
    _compare_inventories(inv1, inv3)


def test_inventory_read_from_json_raises_value_error_for_bad_inputs(
    tmp_path: Path,
) -> None:
    """Test read_from_json raises ValueError for bad paths/strings/schema."""
    inv = HouseholdInventory()

    # Non-existent file path -> treated as JSON string fail -> ValueError
    with pytest.raises(ValueError, match='Error processing input'):
        inv.read_from_json(str(tmp_path / 'does_not_exist.json'))

    # Malformed JSON string -> ValueError
    with pytest.raises(ValueError, match='Error processing input'):
        inv.read_from_json('{not: json}')

    # Schema validation failure -> ValueError (e.g., households not an object)
    bad_schema = {'type': 'HouseholdInventory', 'households': [1, 2, 3]}
    with pytest.raises(ValueError, match='Invalid JSON data'):
        inv.read_from_json(bad_schema)


def test_inventory_read_from_json_raises_file_not_found_when_schema_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test schema missing path raises FileNotFoundError during validation."""
    inv = HouseholdInventory()

    # Keep original Path.open
    orig_open = _Path.open

    def fake_open(self: _Path, *args: object, **kwargs: object) -> object:  # type: ignore[no-redef]
        if str(self).endswith('household_inventory_schema.json'):
            raise FileNotFoundError('schema missing')
        return orig_open(self, *args, **kwargs)

    monkeypatch.setattr(_Path, 'open', fake_open, raising=True)

    with pytest.raises(FileNotFoundError):
        inv.read_from_json({'type': 'HouseholdInventory', 'households': {}})


# test_inventory_round_trip


def test_inventory_round_trip_empty(tmp_path: Path) -> None:
    """Round-trip an empty inventory via JSON file and compare equality."""
    inv = HouseholdInventory()
    path = tmp_path / 'rt_empty.json'
    inv.to_json(str(path))

    loaded = HouseholdInventory()
    loaded.read_from_json(str(path))

    _compare_inventories(inv, loaded)


def test_inventory_round_trip_string_keys(tmp_path: Path) -> None:
    """Round-trip an inventory with string keys and compare equality."""
    inv = HouseholdInventory()
    inv.add_household('H1', Household(features={'a': 1}))
    inv.add_household('H2', Household(features={'b': 2}))

    path = tmp_path / 'rt_str.json'
    inv.to_json(str(path))

    loaded = HouseholdInventory()
    loaded.read_from_json(str(path))

    _compare_inventories(inv, loaded)


def test_inventory_round_trip_int_keys(tmp_path: Path) -> None:
    """Round-trip an inventory with integer keys and compare equality."""
    inv = HouseholdInventory()
    inv.add_household(1, Household(features={'x': 1}))
    inv.add_household(2, Household(features={'y': 2}))

    path = tmp_path / 'rt_int.json'
    inv.to_json(str(path))

    loaded = HouseholdInventory()
    loaded.read_from_json(str(path))

    # Ensure keys came back as ints
    assert set(loaded.inventory.keys()) == {1, 2}
    _compare_inventories(inv, loaded)


# -----------------------------
# 4. Core Logic Tests
# -----------------------------


def test_inventory_get_next_numeric_id_empty() -> None:
    """Test that next numeric ID is 0 when inventory has no numeric keys."""
    inv = HouseholdInventory()
    assert inv._get_next_numeric_id() == 0  # noqa: SLF001 - test intentionally calls private helper


def test_inventory_get_next_numeric_id_non_numeric_only() -> None:
    """Test that next numeric ID is 0 when only non-numeric keys exist."""
    inv = HouseholdInventory()
    inv.add_household('A', Household(features={}))
    inv.add_household('H7X', Household(features={}))
    assert inv._get_next_numeric_id() == 0  # noqa: SLF001 - test intentionally calls private helper


def test_inventory_get_next_numeric_id_numeric_only() -> None:
    """Test that next numeric ID is max(numeric_keys)+1 when only numbers exist."""
    inv = HouseholdInventory()
    inv.add_household(0, Household(features={}))
    inv.add_household(3, Household(features={}))
    inv.add_household(2, Household(features={}))
    # Max numeric key is 3 -> expect 4
    expected_next = 4
    assert inv._get_next_numeric_id() == expected_next  # noqa: SLF001 - test intentionally calls private helper


def test_inventory_get_next_numeric_id_mixed_keys() -> None:
    """Test that next numeric ID ignores non-numeric and increments max numeric."""
    inv = HouseholdInventory()
    inv.add_household('A', Household(features={}))
    inv.add_household(10, Household(features={}))
    inv.add_household('007', Household(features={}))  # numeric string counts as 7
    # Numeric keys present: {10, 7} -> next should be 11
    expected_next = 11
    assert inv._get_next_numeric_id() == expected_next  # noqa: SLF001 - test intentionally calls private helper


# merge_inventory tests


def test_inventory_merge_inventory_no_conflict() -> None:
    """Test merging inventories when no household ID conflicts exist."""
    # Base inventory is empty
    base = HouseholdInventory()

    # Other inventory with two households (string and numeric id)
    other = HouseholdInventory()
    other.add_household('H1', Household(features={'a': 1}))
    other.add_household(2, Household(features={'b': 2}))

    remap = base.merge_inventory(other)

    # IDs should be unchanged and present in base now
    assert remap == {'H1': 'H1', 2: 2}
    assert set(base.inventory.keys()) == {'H1', 2}
    assert base.inventory['H1'].features == {'a': 1}
    assert base.inventory[2].features == {'b': 2}


def test_inventory_merge_inventory_with_conflicts_and_remap() -> None:
    """Test merging with ID conflicts creates new IDs and returns full remap."""
    # Base inventory with one string and one numeric key; next_id starts at 2
    base = HouseholdInventory()
    base.add_household('A', Household(features={'x': 0}))
    base.add_household(1, Household(features={'y': 0}))

    # Other inventory contains conflicting IDs "A" and 1, in this order
    other = HouseholdInventory()
    other.add_household(1, Household(features={'y': 1}))
    other.add_household('A', Household(features={'x': 1}))

    # Merge and get remap
    remap = base.merge_inventory(other)

    # Since conflicts exist, they should be remapped to 2 and 3 respectively
    new_id_1 = 2
    new_id_2 = 3
    assert remap[1] == new_id_1
    assert remap['A'] == new_id_2

    # Validate base now has original plus two new numeric IDs with correct features
    assert set(base.inventory.keys()) == {'A', 1, new_id_1, new_id_2}
    assert base.inventory['A'].features == {'x': 0}
    assert base.inventory[1].features == {'y': 0}
    assert base.inventory[new_id_1].features == {'y': 1}
    assert base.inventory[new_id_2].features == {'x': 1}


def test_inventory_merge_inventory_merge_empty_into_populated() -> None:
    """Test merging an empty inventory into a populated one makes no changes."""
    base = HouseholdInventory()
    base.add_household('H1', Household(features={'a': 1}))
    base.add_household(5, Household(features={'b': 2}))

    other = HouseholdInventory()  # empty

    remap = base.merge_inventory(other)

    # No changes expected
    assert remap == {}
    assert set(base.inventory.keys()) == {'H1', 5}
    assert base.inventory['H1'].features == {'a': 1}
    assert base.inventory[5].features == {'b': 2}


def test_inventory_merge_inventory_raises_type_error_for_non_inventory() -> None:
    """Test that merging with a non-inventory object raises TypeError."""
    base = HouseholdInventory()
    with pytest.raises(TypeError):
        base.merge_inventory(None)  # type: ignore[arg-type]
