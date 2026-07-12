"""Monkey-patch torch.compile to prevent VariableBuilder dispatch errors."""
import torch
import functools

_original_compile = torch.compile

def _patched_compile(*args, **kwargs):
    """No-op compile — just return the function as-is."""
    if len(args) > 0 and callable(args[0]):
        return args[0]
    # Decorator form: return a no-op decorator
    return lambda fn: fn

torch.compile = _patched_compile
torch._dynamo.compile = _patched_compile

if hasattr(torch._dynamo, "optimize"):
    torch._dynamo.optimize = lambda *a, **kw: (lambda fn: fn)

print("✓ torch.compile patched to no-op")
