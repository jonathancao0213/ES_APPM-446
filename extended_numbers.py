import numpy as np

class Q:

    quat = None

    def __init__(self, input):
        self.quat = np.array(input)

    def __repr__(self):
        return "%s + %si + %sj + %sk" % (self.quat[0], self.quat[1], self.quat[2], self.quat[3])

    def __eq__(self, other):
        return np.allclose(self.quat, other.quat)

    def __add__(self, other):
        return Q(np.add(self.quat, other.quat))

    def __sub__(self, other):
        return Q(np.subtract(self.quat, other.quat))
    
    def __mul__(self, other):
        if isinstance(other, Q):
            a, b, c, d = self.quat
            i, j, k, l = other.quat

            return Q(np.array([-b*j - c*k - d*l + a*i,
                            a*j + i*b - k*d + c*l,
                            a*k + i*c - b*l + j*d,
                            a*l + i*d - j*c + b*k]))
        else:
            return Q(np.multiply(self.quat, other))

    def __rmul__(self, other):
        return self.__mul__(other)

class O:
    
    oct = None

    def __init__(self, input):
        self.oct = np.array(input)

    def __repr__(self):
        return "%se0 + %se1 + %se2 + %se3 + %se4 + %se5 + %se6 + %se7" % \
            (self.oct[0], self.oct[1], self.oct[2], self.oct[3], self.oct[4], self.oct[5], self.oct[6], self.oct[7])

    def __eq__(self, other):
        return np.allclose(self.oct, other.oct)
        
    def __add__(self, other):
        return O(np.add(self.oct, other.oct))

    def __sub__(self, other):
        return O(np.subtract(self.oct, other.oct))

    def __mul__(self, other):
        if isinstance(other, O):
            raise Exception("Use O.mul() for multiplying two octonions")
        else:
            return O(np.multiply(self.oct, other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def mul(self,other):
        a,b,c,d,e,f,g,h = self.oct
        i,j,k,l,m,n,o,p = other.oct

        return O(np.array([a*i - b*j - c*k - d*l - e*m - f*n - g*o - h*p,
                            a*j + b*i + c*l - d*k - m*f + n*e + o*h - p*g,
                            a*k - b*l + c*i + d*j - m*g - n*h + o*e + p*f,
                            a*l + b*k - c*j + d*i - m*h + n*g - o*f + p*e,
                            m*a - n*b - o*c - p*d + e*i + f*j + g*k + h*l,
                            m*b + n*a + o*d - p*c - e*j + f*i - g*l + h*k,
                            m*c - n*d + o*a + p*b - e*k + f*l + g*i - h*j,
                            m*d + n*c - o*b + p*a - e*l - f*k + g*j + h*i]))