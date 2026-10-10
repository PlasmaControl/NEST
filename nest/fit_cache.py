"""On-disk cache of the initial fits (``BoundaryPotential.fit``, ``Pressure.fit``).

The fits of the potential take minutes at the default resolution and are
deterministic given the equilibrium file, the arguments and the code, so they
are stored once and loaded afterwards. The key hashes the content of the
equilibrium file, the class, the arguments, and the source of the module that
fits (any change of the code refits). Files go to $NEST_CACHE (default
<repository>/.cache); NEST_CACHE=off disables the cache.
"""

import hashlib
import inspect
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_file_hashes = {}


def _file_hash(path):
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _file_hashes:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _file_hashes[key] = h.hexdigest()
    return _file_hashes[key]


def fit(cls, inp, **kwargs):
    """cls.fit(inp.eq, **kwargs), from the cache when possible (``inp``: DESCInput)."""
    where = os.environ.get("NEST_CACHE", os.path.join(ROOT, ".cache"))
    if where == "off" or getattr(inp, "path", None) is None:
        return cls.fit(inp.eq, **kwargs)
    src = inspect.getsource(inspect.getmodule(cls))
    key = hashlib.sha256(repr((cls.__name__, _file_hash(inp.path), sorted(kwargs.items()),
                               hashlib.sha256(src.encode()).hexdigest())).encode()).hexdigest()
    path = os.path.join(where, f"{cls.__name__}_{key[:24]}.npz")
    if os.path.exists(path):
        return cls.load(path, inp.eq)
    out = cls.fit(inp.eq, **kwargs)
    os.makedirs(where, exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    out.save(tmp)
    os.replace(tmp, path)
    return out
