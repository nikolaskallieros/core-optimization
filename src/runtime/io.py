"""File hashes, JSON records and streamed-download checksums."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def sha(path):
    """Compute a file's SHA-256 without loading the whole file into memory.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input or output file.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def serializable(value):
    """Convert nested NumPy values to ordinary JSON-compatible values.

    Parameters
    ----------
    value : object
        Value to convert or serialize, including nested NumPy containers.

    Returns
    -------
    object
        Converted lists, dictionaries and scalars, without numerical rounding.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def save_json(path, value, *, durable=False):
    """Atomically write a JSON record to its destination.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input or output file.
    value : object
        Value to convert or serialize, including nested NumPy containers.
    durable : bool, optional
        Flush and fsync the temporary file before replacing the destination.

    Raises
    ------
    ValueError
        The payload contains a nonfinite JSON number.
    """
    path = Path(path)
    payload = json.dumps(serializable(value), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_checkpoint(path, value):
    """Replace a Torch checkpoint after writing the complete state.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input or output file.
    value : object
        Value to convert or serialize, including nested NumPy containers.
    """
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class HashStream(io.RawIOBase):
    """Track SHA-256 and byte count while another library consumes a stream."""
    def __init__(self, response, *, progress=None, notify_every=128 * 1024**2):
        """Wrap a readable byte stream with cumulative checksum tracking.

        Parameters
        ----------
        response : binary stream
            Readable source supporting read(count).
        progress : callable or None, optional
            Callback receiving the cumulative byte count after each notification interval.
        notify_every : int, optional
            Minimum number of newly read bytes between progress notifications.
        """
        self.response = response
        self.digest, self.count, self.notified = hashlib.sha256(), 0, 0
        self.progress, self.notify_every = progress, notify_every

    def readable(self):
        """Report that the wrapped stream supports reading.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def readinto(self, buffer):
        """Fill a writable buffer while updating checksum and progress state.

        Parameters
        ----------
        buffer : writable bytes-like object
            Destination buffer whose length limits the requested read.

        Returns
        -------
        int
            Number of bytes copied into the buffer.
        """
        chunk = self.read(len(buffer))
        buffer[:len(chunk)] = chunk
        return len(chunk)

    def read(self, count=-1):
        """Read bytes and update the cumulative SHA-256 and byte count.

        Parameters
        ----------
        count : int, optional
            Maximum bytes to read; a negative value requests all remaining bytes.

        Returns
        -------
        bytes
            Bytes returned by the wrapped stream.
        """
        chunk = self.response.read(count)
        self.digest.update(chunk)
        self.count += len(chunk)
        if self.progress is not None and self.count - self.notified >= self.notify_every:
            self.progress(self.count)
            self.notified = self.count
        return chunk


def save_state(path, state):
    """Write native metadata and compressed arrays as a checksummed pair.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input or output file.
    state : dict
        Nested native state; large numeric arrays move to the adjacent NPZ file while design
        identities remain JSON metadata.

    Raises
    ------
    ValueError
        The state contains nonfinite numerical data.

    Notes
    -----
    Large numerical fields go to the adjacent NPZ; design identities remain JSON metadata.
    Keep both files together when copying retained states.
    """
    path = Path(path)
    arrays = {}

    def encode(value, key="root", design_metadata=False):
        """Move large numerical fields into the companion array collection.

        Parameters
        ----------
        value : object
            Value to convert or serialize, including nested NumPy containers.
        key : str, optional
            Hierarchical identity used as the NPZ array key.
        design_metadata : bool, optional
            Keep numerical design identities inline instead of extracting arrays.

        Returns
        -------
        object
            JSON-compatible subtree with references to extracted arrays.
        """
        if isinstance(value, dict):
            return {name: encode(item, f"{key}/{name}", design_metadata or name == "design")
                    for name, item in value.items()}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (tuple, list, np.ndarray)):
            try:
                array = np.asarray(value)
            except ValueError:
                array = None
            if array is not None and array.ndim == 0:
                return array.item()
            if array is not None and array.dtype.kind in "biuf" and array.size >= 16 and not design_metadata:
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite native state data: {key}")
                arrays[key] = array
                return {"array": key}
            return [encode(item, f"{key}/{i}", design_metadata) for i, item in enumerate(value)]
        return value

    metadata = encode(state)
    array_path = path.with_suffix(".npz")
    temporary_array = array_path.with_suffix(".npz.tmp")
    with temporary_array.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    record = {"format": "native-state-1", "array_file": array_path.name,
              "array_sha256": sha(temporary_array), "state": metadata}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary_array.replace(array_path)
    temporary.replace(path)


def load_state(path):
    """Verify and restore a native JSON/NPZ state pair.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input or output file.

    Returns
    -------
    dict
        Restored metadata and independent NumPy arrays, loaded without pickle.

    Raises
    ------
    ValueError
        The format is unsupported or the array checksum differs from the metadata.
    """
    path = Path(path)
    record = json.loads(path.read_text())
    # Earlier writers prefixed the same v1 schema with their producer name.
    if record["format"] != "native-state-1" and not record["format"].endswith("-native-state-1"):
        raise ValueError("Unknown native state format")
    array_path = path.with_name(record["array_file"])
    if sha(array_path) != record["array_sha256"]:
        raise ValueError("Native state arrays differ from their retained metadata")
    with np.load(array_path, allow_pickle=False) as arrays:
        def decode(value, design_metadata=False):
            """Restore array references from the verified companion NPZ file.

            Parameters
            ----------
            value : object
                Value to convert or serialize, including nested NumPy containers.
            design_metadata : bool, optional
                Restore legacy design arrays as ordinary lists.

            Returns
            -------
            object
                Decoded subtree with copied arrays or design metadata lists.
            """
            if isinstance(value, dict):
                if set(value) == {"array"}:
                    array = arrays[value["array"]]
                    # Design identities are JSON metadata, including legacy long vectors.
                    return array.tolist() if design_metadata else array.copy()
                return {name: decode(item, design_metadata or name == "design") for name, item in value.items()}
            if isinstance(value, list):
                return [decode(item, design_metadata) for item in value]
            return value
        return decode(record["state"])
