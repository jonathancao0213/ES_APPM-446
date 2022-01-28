
import numpy as np
from scipy.fft import fft, ifft, rfft, irfft
from math import floor, ceil

class Basis:

    def __init__(self, N, interval):
        self.N = N
        self.interval = interval


class Fourier(Basis):

    def __init__(self, N, interval=(0, 2*np.pi)):
        super().__init__(N, interval)

    def grid(self, scale=1):
        N_grid = int(np.ceil(self.N*scale))
        return np.linspace(self.interval[0], self.interval[1], num=N_grid, endpoint=False)

    def transform_to_grid(self, data, axis, dtype, scale=1):
        if dtype == np.complex128:
            return self._transform_to_grid_complex(data, axis, scale)
        elif dtype == np.float64:
            return self._transform_to_grid_real(data, axis, scale)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def transform_to_coeff(self, data, axis, dtype):
        if dtype == np.complex128:
            return self._transform_to_coeff_complex(data, axis)
        elif dtype == np.float64:
            return self._transform_to_coeff_real(data, axis)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def _transform_to_grid_complex(self, data, axis, scale):
        # N = len(data)
        # if scale == 1:
        #     return ifft(data, axis=axis)
        # else:
        #     new_data = np.zeros(scale*N, dtype=data.dtype)
        #     if N % 2 == 0:
        #         first = data[:int(N/2)]
        #         last = data[int(N/2):]
        #     else:
        #         first = data[:int((N+1)/2)]
        #         last = data[int((N+1)/2):]

        #     new_data[:len(first)] = first
        #     new_data[-len(last):] = last
        #     return scale*ifft(new_data)

        added = self.N*(scale - 1)
        data = np.insert(data, int(len(data)/2), np.zeros(int(added), dtype=data.dtype))
        return ifft(data)*len(data)


    def _transform_to_coeff_complex(self, data, axis):
        x = fft(data)
        x = np.append(x[0:int(self.N/2)], x[-int(self.N/2):])
        x[-int(self.N/2)] = 0
        return x/len(data)

    def _transform_to_grid_real(self, data, axis, scale):
        complex_data = np.zeros(int(scale*(len(data)//2) + 1), dtype=np.complex128)
        complex_data[:len(data)//2].real = data[::2]
        complex_data[:len(data)//2].imag = data[1::2]
        return irfft(complex_data)*(scale*len(data)//2)
                
    def _transform_to_coeff_real(self, data, axis):
        coeffs = rfft(data)
        coeff_data = np.zeros(len(coeffs)*2)
        coeff_data[::2] = coeffs.real
        coeff_data[1::2] = coeffs.imag
        coeff_data[1] = 0
        # return coeff_data[:self.N]
        return coeff_data[:self.N]/(len(data)//2)

"""
    def _transform_to_grid_real(self, data, axis, scale):
        # added = len(data)*(scale - 1)
        # data = np.append(data, np.zeros(added))
        # complex_data = np.zeros(scale*(len(data)//2+1), dtype=np.complex128)
        # complex_data[:len(data[::2])].real = data[::2]
        # complex_data[:len(data[1::2])].imag = data[1::2]
        # return irfft(complex_data)*(len(data)//2)

        complex_data = np.zeros(scale*len(data), dtype=np.complex128)
        complex_data.real = data[::2]
        complex_data.imag = data[1::2]
        return irfft(complex_data)
            

    def _transform_to_coeff_real(self, data, axis):
        # coeffs = rfft(data)
        # coeff_data = np.zeros(len(data))
        # coeff_data[::2] = coeffs.real[:int(len(data)//2)]
        # coeff_data[1::2] = coeffs.imag[:int(len(data)//2)]
        # coeff_data[1] = 0
        # coeff_data = coeff_data[:self.N//2]
        # return coeff_data/(len(data)//2)

        coeffs = rfft(data)
        coeff_data = np.zeros(len(data))
        coeff_data[::2] = coeffs.real[:int(len(data)//2)]
        coeff_data[1::2] = coeffs.imag[:int(len(data)//2)]
        coeff_data[1] = 0
        return coeff_data
"""


class Domain:

    def __init__(self, bases):
        if isinstance(bases, Basis):
            # passed single basis
            self.bases = (bases, )
        else:
            self.bases = tuple(bases)
        self.dim = len(self.bases)

    @property
    def coeff_shape(self):
        return [basis.N for basis in self.bases]

    def remedy_scales(self, scales):
        if scales is None:
            scales = 1
        if not hasattr(scales, "__len__"):
            scales = [scales] * self.dim
        return scales


class Field:

    def __init__(self, domain, dtype=np.float64):
        self.domain = domain
        self.dtype = dtype
        self.data = np.zeros(domain.coeff_shape, dtype=dtype)
        self.coeff = np.array([True]*self.data.ndim)

    def towards_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        axis = np.where(self.coeff == False)[0][0]
        self.data = self.domain.bases[axis].transform_to_coeff(self.data, axis, self.dtype)
        self.coeff[axis] = True

    def require_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        else:
            self.towards_coeff_space()
            self.require_coeff_space()

    def towards_grid_space(self, scales=None):
        if not self.coeff.any():
            # already in full grid space
            return
        axis = np.where(self.coeff == True)[0][-1]
        scales = self.domain.remedy_scales(scales)
        self.data = self.domain.bases[axis].transform_to_grid(self.data, axis, self.dtype, scale=scales[axis])
        self.coeff[axis] = False

    def require_grid_space(self, scales=None):
        if not self.coeff.any(): 
            # already in full grid space
            return
        else:
            self.towards_grid_space(scales)
            self.require_grid_space(scales)



