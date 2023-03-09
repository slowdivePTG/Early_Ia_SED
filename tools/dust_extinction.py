'''
Extinction models
'''

from lib2to3.pgen2.token import OP
import numpy as np
from pyparsing import Opt
from scipy import interpolate

def fm90( wv, c1, c2, c3, c4, x0, gamma ):
    '''
    Ref: Fitzpatrick & Massa et al. 1990
    Input: 
        wv     - wavelength in Angstrom
        c1     - intercept
        c2     - slope
        c3     - bump strength
        c4     - FUV curvature
        x0     - bump position
        gamma  - bump width
    Return:
        E(wv-V)/E(B-V)
    '''

    def F(x):
        return ( 0.5392 + 0.05644 * ( x - 5.9 ) ) * ( x - 5.9 ) * ( x - 5.9 ) * ( x > 5.9 )

    x = 1.0e4 / wv

    return c1 + c2 * x + \
        c3 * x * x / ( ( x * x - x0 * x0 ) * ( x * x - x0 * x0 ) + x * x * gamma * gamma ) + \
        c4 * F(x)

def ftz( wv, RV ):
    '''
    Ref: Fitzpatrick 1999
    Input: 
        wv     - wavelength in Angstrom
        RV     - R_V
        EBV    - E(B-V)
    Return:
        E(wv-V)/E(B-V)
    '''

    wv = np.float64(wv)

    # UV, lambda < 2700
    c2 = -0.824 + 4.717 / RV
    c1 = 2.030 - 3.007 * c2
    x0 = 4.596
    gamma = 0.99
    c3 = 3.23
    c4 = 0.41
    UVFunc = lambda wv: fm90( wv, c1, c2, c3, c4, x0, gamma )

    # Optical, UV, lambda > 2700
    ## Anchor points
    x = np.array ( [ 0.000, 0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846 ] )
    y = np.array ( [ 0.000, 0.265 * RV / 3.1, 0.829 * RV / 3.1,
                     -0.426 + 1.004 * RV, -0.050 + 1.0016 * RV, 0.701 + 1.0016 * RV,
                     1.208 + 1.0032 * RV - 0.00033 * RV * RV,
                     UVFunc( 2700 ) + RV, UVFunc( 2600 ) + RV ] )
    tck = interpolate.splrep( x, y )
    OptIRFunc = lambda wv: interpolate.splev( 1.0e4 / wv, tck ) - RV

    return np.piecewise( wv, [ wv < 2700, wv >= 2700 ], [ UVFunc, OptIRFunc ] )


def calALambda( wv, RV, EBV, model = 'ftz' ):
    '''
    Calculate A(lambda)
    Input:
        wv      - wavelength in Angstrom
        RV      - R_V
        EBV     - E(B-V)
        model   - extinction model: ftz (Fitzpatrick 1999)
    '''

    if model == 'ftz':
        return ( ftz( wv, RV ) + RV ) * EBV
