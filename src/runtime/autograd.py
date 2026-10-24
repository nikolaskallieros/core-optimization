"""Lossless CPU storage for long autograd histories; no numerical rounding."""
import ctypes
import gc
import zlib

import numpy as np
import torch


class _Saved:
    def __init__(self, value, owner, minimum_bytes):
        """Retain an autograd tensor, compressing CPU bytes when useful.

        Parameters
        ----------
        value : torch.Tensor
            Tensor saved for backward; compression preserves every stored bit.
        owner : GradientStorage
            Owner whose retained-byte counters track this saved value.
        minimum_bytes : int
            Minimum logical tensor size at which compression is attempted.
        """
        value = value.detach()
        self.dtype, self.shape, self.owner = value.dtype, value.shape, owner
        self.logical_bytes = value.numel() * value.element_size()
        self.compressed = False
        self.value = value
        self.stored_bytes = self.logical_bytes
        if value.device.type == "cpu" and value.layout == torch.strided and self.logical_bytes >= minimum_bytes:
            raw = value.resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            packed = zlib.compress(raw, level=1)
            if len(packed) < .9 * len(raw):
                self.value, self.compressed = packed, True
                self.stored_bytes = len(packed)
        owner.logical_bytes += self.logical_bytes
        owner.stored_bytes += self.stored_bytes
        owner.peak_stored_bytes = max(owner.peak_stored_bytes, owner.stored_bytes)

    def unpack(self):
        """Reconstruct the saved tensor without numerical rounding.

        Returns
        -------
        torch.Tensor
            Saved values with their original shape and dtype.
        """
        if not self.compressed:
            return self.value
        raw = np.frombuffer(zlib.decompress(self.value), dtype=np.uint8).copy()
        return torch.from_numpy(raw).view(self.dtype).reshape(self.shape)

    def __del__(self):
        """Release this value's contribution to the owner's byte counters."""
        self.owner.logical_bytes -= self.logical_bytes
        self.owner.stored_bytes -= self.stored_bytes


class GradientStorage:
    """Compress saved CPU autograd tensors losslessly until backward needs them.

    Use context() around the forward calculation; statistics() reports retained
    storage, without changing tensor precision or the derivative graph."""
    def __init__(self, minimum_bytes=32768):
        """Initialize lossless saved-tensor storage and byte counters.

        Parameters
        ----------
        minimum_bytes : int, optional
            Smallest logical tensor size, in bytes, eligible for compression.
        """
        self.minimum_bytes = minimum_bytes
        self.logical_bytes = self.stored_bytes = self.peak_stored_bytes = 0

    def pack(self, tensor):
        """Wrap one saved tensor and update retained-storage statistics.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor retained by an autograd operation for backward.

        Returns
        -------
        _Saved
            Lossless saved representation associated with this storage tracker.
        """
        return _Saved(tensor, self, self.minimum_bytes)

    def context(self):
        """Install saved-tensor hooks around a forward calculation.

        Returns
        -------
        context manager
            Torch hook context that packs and unpacks saved tensors through this tracker.
        """
        return torch.autograd.graph.saved_tensors_hooks(self.pack, lambda value: value.unpack())

    def statistics(self):
        """Report live and peak retained storage in binary megabytes.

        Returns
        -------
        dict
            Logical, stored and peak stored MiB plus the lossless encoding description.
        """
        return {"logical_MiB": self.logical_bytes / 2**20, "stored_MiB": self.stored_bytes / 2**20,
                "peak_stored_MiB": self.peak_stored_bytes / 2**20, "encoding": "lossless zlib level 1"}

    @staticmethod
    def release_unused():
        """Collect unreachable graphs and trim native heaps when supported."""
        gc.collect()
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:
            trim(0)
