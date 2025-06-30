import joblib
from io import BytesIO

import numpy as np
from typing import Tuple, Optional, Any


__all__ = [
    "encode_string_to_bytes",
    "decode_string_from_bytes",
    "encode_u32_to_raw_bytes",
    "decode_u32_from_raw_bytes",
    "encode_float_to_raw_bytes",
    "decode_float_from_bytes",
    "encode_numpy_array_to_raw_bytes",
    "decode_numpy_array_from_bytes",
    "encode_joblib_object_to_bytes",
    "decode_joblib_object_from_bytes"
]

# Specifies how many bytes are used to store the length prefix for encoded strings.
# With 2 bytes, we can represent string lengths up to 65,535 bytes (uint16 range).
LEN_PREFIX_BYTES = 2


def encode_string_to_bytes(s: str) -> bytes:
    """
    Encodes a string into a byte sequence with a 2-byte little-endian length prefix.

    Args:
        s (str): The string to encode.

    Returns:
        bytes: The encoded byte sequence.

    Raises:
        ValueError: If the string exceeds the maximum encodable length.
    """

    s_bytes = s.encode("utf-8")
    s_len = len(s_bytes)
    if s_len > (2 ** (8 * LEN_PREFIX_BYTES) - 1):  # Check if length fits in prefix
        raise ValueError(
            f"String length {s_len} exceeds max for {LEN_PREFIX_BYTES} bytes prefix."
        )
    return s_len.to_bytes(LEN_PREFIX_BYTES, "little") + s_bytes


def decode_string_from_bytes(data: bytes, offset: int) -> Tuple[str, int]:
    """
    Decodes a length-prefixed UTF-8 string from a byte buffer.

    Args:
        data (bytes): The raw byte buffer.
        offset (int): The offset at which the string starts.

    Returns:
        Tuple[str, int]: The decoded string and the new offset.
    """

    s_len = int.from_bytes(data[offset : offset + LEN_PREFIX_BYTES], "little")
    offset += LEN_PREFIX_BYTES
    s = data[offset : offset + s_len].decode("utf-8")
    offset += s_len
    return s, offset

def encode_u32_to_raw_bytes(i: int) -> bytes:
    """
    Encodes a 32-bit unsigned integer into 4 little-endian bytes.

    Args:
        i (int): The integer to encode.

    Returns:
        bytes: A 4-byte sequence representing the integer.
    """

    return i.to_bytes(4, "little", signed=False)

def decode_u32_from_raw_bytes(data: bytes) -> int:
    """
    Decodes a 4-byte little-endian sequence into a 32-bit unsigned integer.

    Args:
        data (bytes): The raw 4-byte buffer.

    Returns:
        int: The decoded integer.
    """

    return int.from_bytes(data, "little", signed=False)

def encode_float_to_raw_bytes(f: float) -> bytes:
    """
    Encodes a float into an 8-byte IEEE 754 float64 little-endian sequence.

    Args:
        f (float): The float to encode.

    Returns:
        bytes: The 8-byte encoded float.
    """

    return np.array([f], dtype=np.float64).tobytes()

def decode_float_from_bytes(data: bytes) -> float:
    """
    Decodes an 8-byte sequence into a float64.

    Args:
        data (bytes): The raw float64 buffer.

    Returns:
        float: The decoded float value.
    """

    return np.frombuffer(data, dtype=np.float64)[0]

def encode_numpy_array_to_raw_bytes(
    arr: np.ndarray,
    as_type: Optional[np.dtype] = np.float64,
) -> bytes:
    """
    Serializes a NumPy array into a raw byte sequence.

    Args:
        arr (np.ndarray): The array to serialize.
        as_type (Optional[np.dtype], optional): If specified, casts the array
            to this dtype before serialization.

    Returns:
        bytes: A contiguous byte buffer of the array.
    """

    if as_type is not None and arr.dtype != as_type:
        arr = arr.astype(as_type, copy=False)
    return np.ascontiguousarray(arr).tobytes()

def decode_numpy_array_from_bytes(
    data: bytes, dtype: np.dtype, shape: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """
    Decodes a byte buffer into a NumPy array.

    Args:
        data (bytes): The raw byte sequence.
        dtype (np.dtype): The dtype to interpret the buffer as.
        shape (Optional[Tuple[int, ...]], optional): If provided, reshapes
            the array to this shape.

    Returns:
        np.ndarray: The decoded NumPy array.
    """

    arr = np.frombuffer(data, dtype=dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def encode_joblib_object_to_bytes(obj: Any) -> bytes:
    """
    Serializes a Python object into bytes using joblib.

    Args:
        obj (Any): The object to serialize.

    Returns:
        bytes: A byte-encoded version of the object.
    """

    buffer = BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()


def decode_joblib_object_from_bytes(data: bytes) -> Any:
    """
    Deserializes a Python object from a joblib-encoded byte sequence.

    Args:
        data (bytes): The raw joblib byte buffer.

    Returns:
        Any: The reconstructed object.
    """
    
    return joblib.load(BytesIO(data))
