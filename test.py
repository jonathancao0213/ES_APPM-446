import numpy as np
import spectral
from scipy import sparse
import pytest
from equations import KdVEquation
import matplotlib.pyplot as plt


KdV_IC1_errors = {16: 0.5, 32: 2e-4, 64: 1e-6}
dtype = np.float64
N = 16
x_basis = spectral.Fourier(N, interval=(0, 4*np.pi))
print(x_basis.wavenumbers(dtype=dtype))
domain = spectral.Domain([x_basis])
x = x_basis.grid()
u = spectral.Field(domain, dtype=dtype)
u.require_grid_space()
u.data = -2*np.cosh((x-2*np.pi))**(-2)

KdV = KdVEquation(domain, u)

KdV.evolve(spectral.SBDF2, 1e-3, 10000)

u.require_coeff_space()
u.require_grid_space(scales=128//N)

sol = np.loadtxt('KdV_IC1.dat')

error = np.max(np.abs(sol - u.data))
print(error)

print(error < KdV_IC1_errors[N])
