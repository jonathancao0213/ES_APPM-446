import pytest
from extended_numbers import O

def test_octonion_eOuality():
    a = O((1,2,3,4,5,6,7,8))
    assert a == O((1,2,3,4,5,6,7,8))

def test_octonion_noneOuality():
    a = O( (1.4, 2.0, -1.4, 0, 1, 1, 1, 1) )
    assert a != O( (1.4, 2.0, 1.4, 0, 1, 1, 1, 2) )

def test_octonion_addition():
    a = O( (1, 0, 3, 2, 1, 2, 3, 4) )
    b = O( (2.4, -1, 7, 0, 4, 3, 2, 1) )
    c = O( (3.4, -1, 10, 2, 5, 5, 5, 5) )
    assert a+b == c

def test_octonion_subtraction():
    a = O( (3.45, -6.459, 3.2, -1.3, 4, 3, 2, 1) )
    b = O( (0, 1.493, 6.3, 1.4, 1, 2, 3, 4) )
    c = O( (3.45, -7.952, -3.1, -2.7, 3, 1, -1, -3) )
    assert a-b == c

def test_octonion_const_mul():
    a = 2.5
    b = O( (1.2, -1, 5.2, 1.35, 1, 2, 3, 4) )
    c = O( (3, -2.5, 13, 3.375, 2.5, 5, 7.5, 10) )
    assert a*b == c

def test_octonion_const_rmul():
    a = 2.5
    b = O( (1.2, -1, 5.2, 1.35, 1, 2, 3, 4) )
    c = O( (3, -2.5, 13, 3.375, 2.5, 5, 7.5, 10) )
    assert b*a == c

def test_octonion_mul_scalar():
    a = O( (1., 0, 0, 0, 0,0,0,0) )
    b = O( (4., 0, 0, 0,0,0,0,0) )
    c = O( (4., 0, 0, 0,0,0,0,0) )
    assert a.mul(b) == c

def test_octonion_mul_double_vector():
    a = O( (0, 3.2, 0, 0, 0,0,0,0) )
    b = O( (0, -4, 0, 0,0,0,0,0) )
    c = O( (12.8, 0, 0, 0,0,0,0,0) )
    assert a.mul(b) == c

def test_octonion_mul_vector():
    a = O((1,2,3,4,5,6,7,8))
    b = O((8,7,6,5,4,3,2,1))
    c = O((-104, 14, 12, 10, 152, 42, 4, 74))
    assert a.mul(b) == c