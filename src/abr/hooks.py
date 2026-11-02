"""Forward hooks: residual-stream capture and directional ablation (no weight edits)."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .common import get_layers


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


@contextmanager
def capture_residual(model, store: dict):
    """store[l] = residual stream at the output of decoder block l, shape [B, T, d] (on CPU, fp32 later)."""
    handles = []
    for l, layer in enumerate(get_layers(model)):
        def hook(_m, _inp, out, l=l):
            store[l] = _hidden(out).detach()
        handles.append(layer.register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def ablate_direction(model, direction: torch.Tensor | None):
    """Project `direction` out of the residual stream at the input and output of every decoder block.

    x <- x - (x . r) r   with r unit-norm, applied at all token positions (prompt and generated).
    """
    if direction is None:
        yield
        return
    r = direction.float()
    r = r / r.norm()
    cache: dict = {}

    def proj_out(x):
        key = (x.device, x.dtype)
        if key not in cache:
            cache[key] = r.to(device=x.device, dtype=x.dtype)
        v = cache[key]
        return x - (x @ v).unsqueeze(-1) * v

    def pre_hook(_m, args, kwargs):
        if args:
            return (proj_out(args[0]),) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = proj_out(kwargs["hidden_states"])
        return args, kwargs

    def post_hook(_m, _inp, out):
        if isinstance(out, tuple):
            return (proj_out(out[0]),) + tuple(out[1:])
        return proj_out(out)

    handles = []
    for layer in get_layers(model):
        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(layer.register_forward_hook(post_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
