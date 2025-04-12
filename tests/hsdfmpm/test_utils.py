import json

import numpy as np
import pytest
from hsdfmpm.utils import iterable_array, read_metadata_json

def test_single_scalar():
    result = iterable_array(5)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] == 5

def test_list_input():
    result = iterable_array([1, 2, 3])
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))

def test_tuple_input():
    result = iterable_array((1, 2))
    np.testing.assert_array_equal(result, np.array([1, 2]))

def test_numpy_array_input():
    arr = np.array([10, 20])
    result = iterable_array(arr)
    # Should return the same values
    np.testing.assert_array_equal(result, arr)
    # But not necessarily the same object (can still be safe)
    assert isinstance(result, np.ndarray)

def test_string_input():
    result = iterable_array("abc")
    # Strings are iterable, so it should return array(['a', 'b', 'c'])
    np.testing.assert_array_equal(result, np.array("abc"))

def test_none_input():
    result = iterable_array(None)
    np.testing.assert_array_equal(result, np.array([None]))

def test_dict_input():
    d = {'a': 1, 'b': 2}
    result = iterable_array(d)
    # Dicts are iterable by keys
    np.testing.assert_array_equal(result, d)

def test_set_input():
    s = {1, 2, 3}
    result = iterable_array(s)
    # Sets are iterable, though unordered
    assert set(result.tolist()) == s


@pytest.fixture
def sample_metadata_file(tmp_path):
    # Create a temporary JSON file with valid metadata
    data = [
        {"AbsTime": 1, "ExpTime": 100, "Filter": "A", "AvgInt": 123.4, "Wavelength": 500},
        {"AbsTime": 2, "ExpTime": 200, "Filter": "B", "AvgInt": 567.8, "Wavelength": 600}
    ]
    file_path = tmp_path / "metadata.json"
    file_path.write_text(json.dumps(data))
    return file_path

def test_read_metadata_json_valid(sample_metadata_file):
    result = read_metadata_json(sample_metadata_file)
    assert result['AbsTime'] == [1, 2]
    assert result['ExpTime'] == [100, 200]
    assert result['Filter'] == ["A", "B"]
    assert result['AvgInt'] == [123.4, 567.8]
    assert result['Wavelength'] == [500, 600]

def test_read_metadata_json_missing_fields(tmp_path):
    data = [
        {"AbsTime": 1, "ExpTime": 100},
        {"Filter": "B", "Wavelength": 600}
    ]
    file_path = tmp_path / "partial.json"
    file_path.write_text(json.dumps(data))

    result = read_metadata_json(file_path)
    assert result['AbsTime'] == [1, None]
    assert result['ExpTime'] == [100, None]
    assert result['Filter'] == [None, "B"]
    assert result['AvgInt'] == [None, None]
    assert result['Wavelength'] == [None, 600]

def test_read_metadata_json_invalid_json(tmp_path, capsys):
    file_path = tmp_path / "bad.json"
    file_path.write_text("{not: valid json}")

    result = read_metadata_json(file_path)
    captured = capsys.readouterr()
    assert "Error decoding JSON" in captured.out
    assert all(v == [] for v in result.values())  # all fields are empty

def test_read_metadata_json_file_not_found(capsys):
    result = read_metadata_json("non_existent_file.json")
    captured = capsys.readouterr()
    assert "File not found" in captured.out
    assert all(v == [] for v in result.values())



if __name__ == '__main__':
    pytest.main()
