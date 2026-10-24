"""Lossless saved tensors and gradients, including noncontiguous views."""
import gc

import pytest
import torch

from runtime.autograd import GradientStorage


@pytest.mark.parametrize('dtype', [torch.float64, torch.int64, torch.bool])
def test_lossless_values_and_repeated_unpack(dtype):
    """Verify lossless values and repeated unpack.

    Parameters
    ----------
    dtype : torch.dtype
        Tensor dtype whose values must survive saved-gradient storage exactly.
    """
    value = torch.zeros((130, 150), dtype=dtype)
    value[4, 8] = True if dtype == torch.bool else 3
    value = value.T[:, ::2]
    storage = GradientStorage(minimum_bytes=1)
    saved = storage.pack(value)
    assert saved.compressed
    assert torch.equal(saved.unpack(), value)
    assert torch.equal(saved.unpack(), value)
    assert torch.equal(saved.unpack().contiguous().view(torch.uint8), value.contiguous().view(torch.uint8))
    assert storage.stored_bytes < storage.logical_bytes / 10
    del saved
    assert storage.logical_bytes == storage.stored_bytes == 0


def test_reverse_derivatives_match_uncompressed_shared_history():
    """Verify reverse derivatives match uncompressed shared history."""
    initial = torch.zeros((130, 150), dtype=torch.float64)
    initial[3, 4] = .7
    results = []
    for compress in (False, True):
        x = initial.clone().requires_grad_()
        scale = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
        storage = GradientStorage(minimum_bytes=1)
        from contextlib import nullcontext
        with storage.context() if compress else nullcontext():
            a = (x * scale).T
            first = a.exp().square().sum()
            second = (a[:, ::2].sin() * a[:, ::2]).sum()
            value = first + second
            results.append((value.detach(), torch.autograd.grad(value, (x, scale))))
        del value, first, second, a
        gc.collect()
        assert storage.logical_bytes == storage.stored_bytes == 0
    assert torch.equal(results[0][0], results[1][0])
    for direct, packed in zip(results[0][1], results[1][1]):
        assert torch.equal(direct, packed)
