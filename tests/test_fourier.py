"""Step 4: FFT of J on the periodic box G3.

  1. wavevectors: a cosine with a known k peaks at that k
  2. FFT of J = curl T (free-boundary precise_QA and W7-X, J.n = 0 on the
     boundary): the inverse returns J,
     Parseval holds (normalization), and J_hat(k = 0) = sum J

Also printed: the fraction of J_hat along k, i.e. of J that is not
divergence-free (removed by the projection in step 5).

run: pytest tests/test_fourier.py -s     or     python tests/test_fourier.py
"""

import os
import sys

import numpy as np

try:  # pytest is optional: the file also runs as a plain script
    import pytest
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import desc.io  # noqa: E402

from cartesian_grid import cartesian_grid, current_on_cartesian  # noqa: E402
from current_potential import CurrentPotential  # noqa: E402
from fourier import fft_J, ifft_J, nyquist, wavevectors  # noqa: E402

EQ_FILES = [
    os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_L16N14.h5"),
    os.path.join(ROOT, "desc_input", "W7-X_output.h5"),
]


def rfft_weights(Nz):
    """Weight of each kz plane of an rfft in sums over the full spectrum."""
    w = np.full(Nz // 2 + 1, 2.0)
    w[0] = 1
    if Nz % 2 == 0:
        w[-1] = 1
    return w


def test_wavevectors():
    """cos(k0 . x) on a box: its spectrum is two peaks at +-k0 (one in the rfft half)."""
    axes = (0.3 + 0.01 * np.arange(64), -0.2 + 0.01 * np.arange(48), 0.05 * np.arange(32))
    L = np.array([len(a) * (a[1] - a[0]) for a in axes])
    k0 = 2 * np.pi * np.array([3, -2, 5]) / L
    X = np.meshgrid(*axes, indexing="ij")
    f = np.cos(sum(k0[i] * X[i] for i in range(3)))
    F = np.abs(np.asarray(fft_J(np.stack([f] * 3, axis=-1).astype(np.float32))[..., 0]))
    kx, ky, kz = wavevectors(axes)
    i, j, k = np.unravel_index(np.argmax(F), F.shape)
    print(f"\npeak at k = ({kx[i]:.3f}, {ky[j]:.3f}, {kz[k]:.3f}), "
          f"expected +-({k0[0]:.3f}, {k0[1]:.3f}, {k0[2]:.3f})")
    assert np.allclose(np.abs([kx[i], ky[j], kz[k]]), np.abs(k0))
    assert np.isclose(F.max(), f.size / 2, rtol=1e-4)  # unnormalized forward FFT
    assert np.isclose(nyquist(axes)[0][0], np.pi / 0.01)


def check_fft_of_J(path, frac=10):
    eq = desc.io.load(path)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    h = float(eq.compute("a")["a"]) / frac
    axes = cartesian_grid(eq, h)
    J = current_on_cartesian(eq, CurrentPotential.fit(eq), axes)["J"]
    J_hat = fft_J(J)
    shape = J.shape[:3]
    print(f"\n{os.path.basename(path)}: box {shape}, h = a/{frac}, "
          f"J_hat {J_hat.shape} {J_hat.dtype}")

    # inverse returns J
    err = np.abs(np.asarray(ifft_J(J_hat, shape)) - J).max() / np.abs(J).max()
    # Parseval: sum |J|^2 = (1/N) sum over the full spectrum of |J_hat|^2
    w = rfft_weights(shape[2])
    P = np.abs(np.asarray(J_hat, dtype=np.complex128)) ** 2
    lhs = np.sum(J.astype(np.float64) ** 2)
    rhs = np.einsum("abcd,c->", P, w) / np.prod(shape)
    # k = 0 mode
    dc = np.asarray(J_hat[0, 0, 0]).real
    sumJ = J.sum(axis=(0, 1, 2), dtype=np.float64)

    # fraction of J_hat along k (not divergence-free), k != 0
    kx, ky, kz = wavevectors(axes)
    K = np.stack(np.meshgrid(kx, ky, kz, indexing="ij"), axis=-1)
    k2 = np.sum(K**2, axis=-1)
    k2[0, 0, 0] = 1
    kJ = np.abs(np.einsum("abci,abci->abc", K, np.asarray(J_hat, np.complex128))) ** 2 / k2
    comp = np.einsum("abc,c->", kJ, w) / np.einsum("abcd,c->", P, w)

    print(f"inverse: max err / max|J| = {err:.1e}; Parseval: rel diff "
          f"{abs(lhs - rhs) / lhs:.1e}; J_hat(0) vs sum J: max rel diff "
          f"{np.abs(dc - sumJ).max() / np.abs(J).sum():.1e}")
    print(f"fraction of |J_hat|^2 along k (non-solenoidal part): {comp:.2e}")
    assert err < 1e-5
    assert abs(lhs - rhs) / lhs < 1e-5
    assert np.abs(dc - sumJ).max() / np.abs(J).sum() < 1e-5


try:
    @pytest.mark.parametrize("path", EQ_FILES, ids=[os.path.basename(f) for f in EQ_FILES])
    def test_fft_of_J(path):
        check_fft_of_J(path)
except NameError:
    pass


if __name__ == "__main__":
    test_wavevectors()
    for path in EQ_FILES:
        check_fft_of_J(path)
