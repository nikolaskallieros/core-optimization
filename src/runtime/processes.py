"""Native-library discovery and process cleanup for local calculations."""
from contextlib import contextmanager
import os
from pathlib import Path
import resource
import signal
import sys


@contextmanager
def interrupt_on_sigterm():
    """Temporarily translate SIGTERM into a catchable KeyboardInterrupt.

    Yields
    ------
    None
        Control while the temporary main-thread signal handler is active.

    Notes
    -----
    The previous SIGTERM handler is restored when the context exits, allowing callers to
    save status and close native children during interruption.
    """
    def interrupt(signum, frame):
        """Raise an interrupt when the installed signal handler is invoked.

        Parameters
        ----------
        signum : int
            Received signal number.
        frame : frame or None
            Interrupted Python frame, supplied by the signal module and unused here.

        Raises
        ------
        KeyboardInterrupt
            The process received the intercepted termination signal.
        """
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def peak_rss_mib():
    """Return the process peak resident memory on supported Unix platforms.

    Returns
    -------
    float
        Peak resident set size in MiB, accounting for platform-specific resource units.
    """
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def native_library_path():
    """Locate the configured or environment-provided OpenMC shared library.

    Returns
    -------
    pathlib.Path
        Resolved native library path.

    Raises
    ------
    FileNotFoundError
        No explicit path is configured and no supported OpenMC library is installed.
    """
    import openmc
    configured = os.environ.get("OPENMC_REFERENCE_LIBRARY")
    if configured:
        return Path(configured).expanduser().resolve()
    package = Path(openmc.__file__).resolve().parent
    for directory in (package / "lib", Path(sys.prefix) / "lib", Path(sys.prefix) / "Library/bin"):
        for name in ("libopenmc.dylib", "libopenmc.so", "openmc.dll"):
            path = directory / name
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError("Native OpenMC is missing. Create the environment from environment.yml.")
