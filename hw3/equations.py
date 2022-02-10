
import numpy as np
import spectral
from scipy import sparse

"""
class KdVEquation:

    def __init__(self, domain, u):
        self.dtype = u.dtype
        dtype = self.dtype
        self.u = u
        self.domain = domain
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.RHS = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.RHS], dtype=dtype)
        self.x_basis = domain.bases[0]
        self.kx = self.x_basis.wavenumbers(dtype)

        p = self.problem.pencils[0]

        I = sparse.eye(self.x_basis.N)
        p.M = I
        diag = -self.kx**3
        p.L = sparse.diags(diag)

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        x_basis = self.x_basis
        dudx = self.dudx
        RHS = self.RHS

        for i in range(num_steps):
            u.require_coeff_space()
            dudx.require_coeff_space()
            dudx.data = 1j*x_basis.wavenumbers(self.dtype)*u.data
            u.require_grid_space(scales=3/2)
            dudx.require_grid_space(scales=3/2)
            RHS.require_grid_space(scales=3/2)
            RHS.data = 6*u.data*dudx.data

            ts.step(dt)
"""

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
        diag = x_basis.wavenumbers(dtype)**3
        p.L = sparse.diags(diag)
        
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
            dudx.data = 1j*x_basis.wavenumbers(self.dtype)*u.data
            u.require_grid_space()
            dudx.require_grid_space()
            RHS.require_grid_space()
            RHS.data = 6 * u.data * dudx.data

            # take timestep
            ts.step(dt)

class SHEquation:

    def __init__(self, domain, u):
        pass

    def evolve(self, timestepper, dt, num_steps):
        pass


