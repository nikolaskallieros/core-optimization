"""Scoped NJOY scratch state and durable failure receipts."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from runtime.io import save_json


@contextmanager
def processing_context(directory, receipt):
    """Scope processing scratch settings and retain failure diagnostics.

    Parameters
    ----------
    directory : str or pathlib.Path
        Processing destination containing a scratch subdirectory.
    receipt : pathlib.Path
        Existing processing manifest updated with failure status when an exception occurs.

    Yields
    ------
    None
        Control with scratch-local temporary files and one OpenMP thread.

    Notes
    -----
    Restores the caller's temporary-directory and thread environment even on failure.
    Exceptions are recorded when a receipt exists and then re-raised.
    """
    directory = Path(directory)
    scratch = directory / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    previous_temp = tempfile.tempdir
    previous_threads = os.environ.get("OMP_NUM_THREADS")
    try:
        tempfile.tempdir = str(scratch.resolve())
        os.environ["OMP_NUM_THREADS"] = "1"
        yield
    except Exception as error:
        if receipt.is_file():
            record = json.loads(receipt.read_text())
            record.update(status="failed", pid=None, error=f"{type(error).__name__}: {error}")
            save_json(receipt, record)
        raise
    finally:
        tempfile.tempdir = previous_temp
        if previous_threads is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous_threads
