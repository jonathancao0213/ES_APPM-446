
import numpy as np
import spectral
from scipy import sparse
class KdVEquation:
    
    def __init__(self, domain, u):
        # store data we need for later, make M and L matrices
        self.dtype = u.dtype
        dtype = self.dtype
        self.u = u
        self.domain = domain
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.RHS = spectral.Field(domain, dtype=dtype) # 6*u*dudx
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.RHS], dtype=dtype)

        p = self.problem.pencils[0]

        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        if dtype == np.complex128:
            diag = -1j*x_basis.wavenumbers(dtype)**3
            p.L = sparse.diags(diag)
        else:
            diag = x_basis.wavenumbers(dtype)**3
            d = np.zeros(len(diag))
            d[::2] = diag[1::2]
            d[1::2] = -diag[::2]
            p.L = sparse.diags(d, dtype=dtype)
        
    def evolve(self, timestepper, dt, num_steps): # take timesteps
        ts = timestepper(self.problem)
        x_basis = self.domain.bases[0]
        u = self.u
        dudx = self.dudx
        RHS = self.RHS

        for i in range(num_steps):
            # need to calculate -u*ux and put it into RHS
            u.require_coeff_space()
            dudx.require_coeff_space()
            if self.dtype == np.complex128:
                dudx.data = 1j*x_basis.wavenumbers(self.dtype)*u.data
            else:
                n = np.zeros(len(x_basis.wavenumbers(self.dtype)))
                n[::2] = -x_basis.wavenumbers(self.dtype)[1::2]
                n[1::2] = x_basis.wavenumbers(self.dtype)[::2]
                dudx.data = n*u.data
            u.require_grid_space(scales=3/2)
            dudx.require_grid_space(scales=3/2)
            RHS.require_grid_space(scales=3/2)
            RHS.data = 6 * u.data * dudx.data

            # take timestep
            ts.step(dt)


class SHEquation:

    def __init__(self, domain, u):
        self.dtype = u.dtype
        dtype = self.dtype
        self.u = u
        self.domain = domain
        self.RHS = spectral.Field(domain, dtype=dtype) # 6*u*dudx
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.RHS], dtype=dtype)

        p = self.problem.pencils[0]

        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N, dtype=dtype)
        p.M = I
        if dtype == np.complex128:
            diag = -2*x_basis.wavenumbers(dtype)**2 + x_basis.wavenumbers(dtype)**4
            p.L = sparse.diags(diag)
        else:
            diag2 = x_basis.wavenumbers(dtype)**2
            diag4 = x_basis.wavenumbers(dtype)**4
            d = -2*diag2 + diag4
            p.L = sparse.diags(d, dtype=dtype)

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        x_basis = self.domain.bases[0]
        u = self.u
        RHS = self.RHS

        for i in range(num_steps):
            # need to calculate -u*ux and put it into RHS
            u.require_coeff_space()
            u.require_grid_space(scales=3/2)
            RHS.data = -u.data**3 + 1.8*u.data**2 - 1.3*u.data

            # take timestep
            ts.step(dt)


