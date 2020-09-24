"""Python module polynomials.py

Functions for polynomial function manipulation for discrete linear
dynamical systems modelling and control applications.

Contents:

 - sub_poly(a, b)
 - diophantine(C, D)
 - diophantine_recursive(C, D, hp)

1. Polynomial division by recursive solution of the Diophantine equation

    F, M = diophantine_recursive(C, D, hp) solves the following
    diophantine equation for j = 1 to hp:

          C / D = F + z^(-j) * M / D

    where C, D, F, and M are polynomials in decreasing powers of z^-1.

    Example:

    Solve the following for the 2-step-ahead prediction calculation:

        (1 - 0.2z^-1) / (1 - 0.8z^-1) = F + z^(-j) * M / (1 - 0.8z^-1)

    >>> F, M = diophantine_recursive([1, -0.2], [1, -0.8], 2)

    We obtain the results for 1-step ahead and 2-steps ahead:

    >>> F
    array([[1. , 0. ],
        [1. , 0.6]])
    >>> M
    array([[0.6 ],
        [0.48]])

    Therefore:
        (1 - 0.2z^-1) / (1 - 0.8z^-1) = 1 + 0.6z^-1 / (1 - 0.8z^-1)

    and

        (1 - 0.2z^-1) / (1 - 0.8z^-1)
            = 1 + 0.6z^-1 + 0.48z^-2 / (1 - 0.8z^-1)

"""
# Based on the MATLAB script Diophantine_equation.m by André Desbiens
# and Kaddour Najim, Copyright (c), April 1993.
# Translated to Python by Bill Tubbs, July 2020.

import numpy as np


def add_poly(a, b, align=-1):
    """Returns the subtraction of two polynomials.

        p = a + b 
    
    where the polynomials `p`, `a`, and `b` are 
    represented as lists or arrays containing the 
    coefficients in decreasing powers.

    `a` and `b` do not need to be the same length.

    Parameters:
        a : list or array
            Coefficients of a
        b : list or array
            Coefficients of b
        align (int): 0 or -1
            align=0 : indicates that the first items
            in a and b correspond to terms having the
            same power. align=-1 indicates that the 
            last items are aligned.

    Returns:
        p : array
            Coefficients of p

    Example:
    >>> add_poly([1, 4, 4], [4, 2])
    array([1., 8., 6.])
    >>> add_poly([1, 0.2], [1, 0.1, 0.8], align=0)
    array([2. , 0.3, 0.8])
    """
    a, b = np.array(a), np.array(b)
    la, lb = len(a), len(b)
    if la == lb:
        return a + b
    else:
        p = np.zeros(max((la, lb)))  # Always returns dtype float
        if align == 0:
            p[:la] = a
            p[:lb] = p[:lb] + b
        elif align == -1:
            p[-la:] = a
            p[-lb:] = p[-lb:] + b
        else:
            raise ValueError("Invalid value for align")
        return p


def sub_poly(a, b, align=-1):
    """Returns the subtraction of two polynomials.

        p = a - b 
    
    where the polynomials `p`, `a`, and `b` are 
    represented as lists or arrays containing the 
    coefficients in decreasing powers.

    `a` and `b` do not need to be the same length.

    Parameters:
        a : list or array
            Coefficients of a
        b : list or array
            Coefficients of b
        align (int): 0 or -1
            align=0 : indicates that the first items
            in a and b correspond to terms having the
            same power. align=-1 indicates that the 
            last items are aligned.

    Returns:
        p : array
            Coefficients of p

    Example:
    >>> sub_poly([1, 4, 4], [4, 2])
    array([1., 0., 2.])
    >>> sub_poly([1, 0.2], [1, 0.1, 0.8], align=0)
    array([ 0. ,  0.1, -0.8])
    """
    a, b = np.array(a), np.array(b)
    la, lb = len(a), len(b)
    if la == lb:
        return a - b
    else:
        p = np.zeros(max((la, lb)))  # Always returns dtype float
        if align == 0:
            p[:la] = a
            p[:lb] = p[:lb] - b
        elif align == -1:
            p[-la:] = a
            p[-lb:] = p[-lb:] - b
        else:
            raise ValueError("Invalid value for align")
        return p


def div_poly(C, D):
    """Finds the first quotient and the remainder of a
    polynomial division:
    
        C / D = quotient + remainder / D
    
    where `C`, `D`, `quotient` and `remainder` are 
    polynomials represented as a list or array of 
    coefficients of decreasing powers.

    Note: The last item in each list corresponds to
    the coefficient of order 0 and all coefficients
    between this and the highest order term must be
    included. 

    Example:
        Find the quotient and remainder of the
        following:
            x**2 + 9*x + 20 / (x + 5)

    >>> q, r = div_poly([1, 9, 20], [1, 5])                                
    >>> print(q, r)                                                        
    [1. 0.] [ 4. 20.]

        Therefore:
            x**2 + 9*x + 20 / (x + 5)
                = x + (4*x + 20) / (x + 5)

        To complete the polynomial, division
        call div_poly again on the remainder:

    >>> q2, r = div_poly(r, [1, 5])                                
    >>> print(add_poly(q, q2), r)
    [1. 4.] [0.]

        Therefore:
            x**2 + 9*x + 20 / (x + 5)
                = x + 4
    """
    C = np.array(C)
    D = np.array(D)
    C = C[np.nonzero(C)[0][0]:]  # Remove any leading zeros
    D = D[np.nonzero(D)[0][0]:]  # Remove any leading zeros
    r = len(C) - len(D)   # relative degree of numerator
    assert(r >= 0), "degree of numerator less than denominator"
    quotient = np.zeros(r + 1)
    quotient[0] = C[0] / D[0]
    remainder = sub_poly(C, quotient[0] * D, align=0)
    assert remainder[0] == 0
    return quotient, remainder[1:]


def diophantine(C, D):
    """Solves the following diophantine equation:
    
        C / D = quotient + remainder / D
    
    where `C`, `D`, and `remainder` are polynomials
    represented as a list or array of coefficients
    in decreasing powers of z^-1.
    """
    # TODO: Consider replacing this with div_poly function
    D = np.array(D)
    quotient = C[0] / D[0]
    remainder = sub_poly(C, quotient * D, align=0)
    assert remainder[0] == 0
    return quotient, remainder[1:]


def diophantine_recursive(C, D, hp):
    """Recursive solution of the Diophantine equation.

    Returns the polynomials `F` and `M` that satisfy

        C / D = F + z^(-j) * M / D

    for j = 1 to `hp`, where `C`, `D`, `F`, and `M` are
    polynomials represented as lists or arrays containing
    the coefficients in decreasing powers of z^-1.

    Arguments:
        C : array_like
            Coefficients of `C`.
        D : array_like
            Coefficients of `D`.
        hp : int
            Number of iterations of recursion.

    Returns:
        F : ndarray (2-D)
            Coefficients of `F` in rows for j = 1 to `hp`.
        M : ndarray (2-D)
            Coefficients of `M` in rows for j = 1 to `hp`.

    Example:
    >>> # Solve for 2-steps ahead:
    >>> # (1 - 0.2z^-1) / (1 - 0.8z^-1) = 1 + 0.6z^-1 + 0.48z^-2 / (1 - 0.8z^-1)
    >>> F, M = diophantine_recursive([1, -0.2], [1, -0.8], 2)
    >>> F
    array([[1. , 0. ],
        [1. , 0.6]])
    >>> M
    array([[0.6 ],
        [0.48]])
    """
    C = np.array(C)
    D = np.array(D)
    F = np.zeros((hp, hp))
    M = np.zeros((hp, max(len(C), len(D)) - 1))
    quotients = []
    numerator = C
    for j in range(hp):
        q, r = diophantine(numerator, D)
        quotients.append(q)
        F[j,:j+1] = quotients
        M[j,:] = r
        numerator = r
    
    return F, M


def test_add_poly():
    """Unit test sub_poly (compared to MATLAB outputs)"""
    assert np.array_equal(add_poly([8], [5]), [13])
    assert np.array_equal(add_poly(np.array([8]), np.array([5])), [13])
    a = [1, 2, 3]
    b = [1, 2]
    assert np.array_equal(add_poly(a, a), [2., 4., 6.])
    assert np.array_equal(add_poly(b, b), [2., 4.])
    assert np.array_equal(add_poly(a, b), [1., 3., 5.])
    assert np.array_equal(add_poly(a, b, align=0), [2., 4., 3.])
    assert np.array_equal(add_poly(a, b, align=-1), [1., 3., 5.])
    assert np.array_equal(add_poly(b, a), [1., 3., 5.])
    assert np.array_equal(add_poly(b, a, align=0), [2., 4., 3.])
    a = np.array([0.5, -0.6, 0.7, -0.8])  # dtype: float
    assert np.array_equal(add_poly(a, -a, align=0), [0., 0., 0., 0.])
    assert np.array_equal(add_poly(a, b, align=0), [1.5, 1.4, 0.7, -0.8])
    assert np.array_equal(add_poly(b, a, align=0), [1.5, 1.4, 0.7, -0.8])


def test_sub_poly():
    """Unit test sub_poly (compared to MATLAB outputs)"""
    assert np.array_equal(sub_poly([8], [5]), [3])
    assert np.array_equal(sub_poly(np.array([8]), np.array([5])), [3])
    a = [1, 2, 3]
    b = [1, 2]
    assert np.array_equal(sub_poly(a, a), [0., 0., 0.])
    assert np.array_equal(sub_poly(a, a, align=0), [0., 0., 0.])
    assert np.array_equal(sub_poly(a, a, align=-1), [0., 0., 0.])
    assert np.array_equal(sub_poly(b, b), [0., 0.])
    assert np.array_equal(sub_poly(a, b), [1., 1., 1.])
    assert np.array_equal(sub_poly(a, b, align=0), [0., 0., 3.])
    assert np.array_equal(sub_poly(a, b, align=-1), [1., 1., 1.])
    assert np.array_equal(sub_poly(b, a), [-1., -1., -1.])
    assert np.array_equal(sub_poly(b, a, align=0), [0., 0., -3.])
    a = np.array([0.5, -0.6, 0.7, -0.8])  # dtype: float
    assert np.array_equal(sub_poly(a, a), [0., 0., 0., 0.])
    assert np.array_equal(sub_poly(a, b), [ 0.5, -0.6, 0.7 - 1, -2.8])
    assert np.array_equal(sub_poly(a, b, align=0), [-0.5, -2.6, 0.7, -0.8])
    assert np.array_equal(sub_poly(b, a), [-0.5, 0.6, 1 - 0.7, 2.8])
    assert np.array_equal(sub_poly(b, a, align=0), [0.5, 2.6, -0.7, 0.8])


def test_diophantine():
    A = np.array([0.18, 1.14, 1.43, 0.48, 0.49])
    quotient, remainder = diophantine(A, A)
    assert quotient == 1.0
    assert np.array_equal(remainder, np.zeros(len(A) - 1))
    quotient, remainder = diophantine([0], [1, -0.8])
    assert quotient == 0.0
    assert np.array_equal(remainder, [0.0])
    quotient, remainder = diophantine([4], [1, -0.8])
    assert quotient == 4.0
    assert np.array_equal(remainder, [3.2])


def test_div_poly():
    A = np.array([0.18, 1.14, 1.43, 0.48, 0.49])
    quotient, remainder = div_poly(A, A)
    assert quotient == [1.0]
    assert np.array_equal(remainder, np.zeros(len(A) - 1))
    quotient, remainder = div_poly([0], [1, -0.8])
    assert quotient == [0.0]
    assert np.array_equal(remainder, [0.0])
    quotient, remainder = div_poly([4], [1, -0.8])
    assert quotient == [4.0]
    assert np.array_equal(remainder, [3.2])

    # Example 1 from this video: https://www.youtube.com/watch?v=8lT00iLntFc
    C, D = [1, 9, 20], [1, 5]
    q, r = div_poly(C, D)
    assert np.array_equal(q, np.array([1., 0.]))
    assert np.array_equal(r, np.array([4., 20.]))
    q2, r = div_poly(r, D)
    assert np.array_equal(q2, np.array([4.]))
    assert np.array_equal(r, np.array([0.]))
    
    # Example 3 from this video: https://www.youtube.com/watch?v=8lT00iLntFc
    C, D = [3, 0, 4, 0, -5, 8], [1, 0, 3]
    q, r = div_poly(C, D)
    assert np.array_equal(q, np.array([3., 0., 0., 0.]))
    assert np.array_equal(r, np.array([ 0., -5., 0., -5., 8.]))
    q2, r = div_poly(r, D)
    assert np.array_equal(q2, np.array([-5., 0.]))
    assert np.array_equal(r, np.array([ 0., 10., 8.]))


def test_diophantine_recursive():
    """Unit test diophantine_recursive (compared to MATLAB outputs)"""
    # Test case 1
    F, M = diophantine_recursive([1], [2], 1)  
    assert np.array_equal(F, np.array([[0.5]]))
    assert np.array_equal(M, np.array([[]]))
    # Test case 2
    C, D, hp = [1, -0.2], [1, -0.8], 2
    F, M = diophantine_recursive(C, D, hp)
    F_test = np.array([
        [1.0, 0.0],
        [1.0, 0.6]
    ])
    M_test = np.array([
        [0.60],
        [0.48]
    ])
    assert np.allclose(F, F_test)
    assert np.allclose(M, M_test)
    # Test case 3
    C, D, hp = [1], [1, -1.8, 0.8], 3
    F, M = diophantine_recursive(C, D, hp)
    F_test = np.array([
        [1.0000,         0,         0],
        [1.0000,    1.8000,         0],
        [1.0000,    1.8000,    2.4400]
    ])
    M_test = np.array([
        [1.8000,   -0.8000],
        [2.4400,   -1.4400],
        [2.9520,   -1.9520]
    ])
    assert np.allclose(F, F_test)
    assert np.allclose(M, M_test)
    # Test case 4
    C, D, hp = [1, 0.6, -0.2], [1, 0.6, -0.2], 3
    F, M = diophantine_recursive(C, D, hp)
    F_test = np.array([
        [1., 0. , 0. ],
        [1., 0. , 0. ],
        [1., 0. , 0. ]
    ])
    M_test = np.zeros((3, 2))
    assert np.allclose(F, F_test)
    assert np.allclose(M, M_test)


if __name__ == '__main__':
    # Run unit tests
    test_sub_poly()
    test_diophantine()
    test_diophantine_recursive()
