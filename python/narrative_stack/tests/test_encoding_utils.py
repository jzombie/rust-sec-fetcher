import numpy as np

from us_gaap_store.binary_codec import (
    encode_string_to_bytes,
    decode_string_from_bytes,
    encode_u32_to_raw_bytes,
    decode_u32_from_raw_bytes,
    encode_float_to_raw_bytes,
    decode_float_from_bytes,
    encode_numpy_array_to_raw_bytes,
    decode_numpy_array_from_bytes,
    encode_joblib_object_to_bytes,
    decode_joblib_object_from_bytes,
)

def test_encode_decode_string():
    s = "hello world"
    encoded = encode_string_to_bytes(s)
    decoded, offset = decode_string_from_bytes(encoded, 0)
    assert decoded == s
    assert offset == len(encoded)

def test_encode_decode_u32():
    i = 4294967295  # max u32
    encoded = encode_u32_to_raw_bytes(i)
    decoded = decode_u32_from_raw_bytes(encoded)
    assert decoded == i

def test_encode_decode_float():
    f = 3.141592653589793
    encoded = encode_float_to_raw_bytes(f)
    decoded = decode_float_from_bytes(encoded)
    assert np.isclose(decoded, f)

def test_encode_decode_numpy_array():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
    encoded = encode_numpy_array_to_raw_bytes(arr, as_type=np.uint32)
    decoded = decode_numpy_array_from_bytes(encoded, dtype=np.uint32, shape=(2, 3))
    np.testing.assert_array_equal(decoded, arr)

def test_encode_decode_numpy_array_cast():
    arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    encoded = encode_numpy_array_to_raw_bytes(arr, as_type=np.float32)
    decoded = decode_numpy_array_from_bytes(encoded, dtype=np.float32)
    np.testing.assert_array_almost_equal(decoded, arr.astype(np.float32))

def test_encode_decode_joblib_object():
    obj = {"a": 1, "b": [2, 3, 4], "c": (5, {"x": 6})}
    encoded = encode_joblib_object_to_bytes(obj)
    decoded = decode_joblib_object_from_bytes(encoded)
    assert decoded == obj
