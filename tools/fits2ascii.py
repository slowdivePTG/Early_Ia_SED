from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def wavelength_convert_air_vacuum(lambda1, verbose=0):
    '''
    lambda_air = lambda_vac / n; where n is tmospheric refractivity

    n = 1 + 6.4328e-5 + (2.94981e6)/(1.46e10-sigma**2) + (2.5540e4)/(4.1e9-sigma**2); (old) where sigma is wave number

    n = 1 + 8.34213e-5 + (2.406030e6)/(1.30e10-sigma**2) + (1.5997e4)/(3.89e9-sigma**2)

    '''

    sigma = 1.0/lambda1*1e8

    #n = 1 + 6.4328e-5 + (2.94981e6)/(1.46e10-sigma**2) + (2.5540e4)/(4.1e9-sigma**2)
    n = 1 + 8.34213e-5 + (2.406030e6)/(1.30e10-sigma**2) + \
        (1.5997e4)/(3.89e9-sigma**2)

    lambda2 = lambda1/n

    if verbose:
        print("The atmospheric refractivity at %s angstrom is %s" % (lambda1, n))

    return lambda2

def fits2ascii(file, output, lower=-np.inf, upper=np.inf):
    sn = fits.open(file)
    header = (sn[0].header + sn[1].header).tostring(sep="\n", endcard=False,padding=False)
    wv = wavelength_convert_air_vacuum(sn[1].data["wave"])
    fl = sn[1].data['flux']
    unc = sn[1].data["ivar"] ** -0.5
    fl = fl[(wv > lower) & (wv < upper)]
    unc = unc[(wv > lower) & (wv < upper)]
    wv = wv[(wv > lower) & (wv < upper)]
    np.savetxt(
          output,
          np.array([wv, fl, unc]).T,
          fmt=("%.4f", "%.4f", "%.4e"),
          header=header,
      )
    plt.plot(wv, fl)
    plt.show()
