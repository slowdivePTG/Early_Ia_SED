import numpy as np
from jax import lax


def spec_to_flux(spec, flt, cov=[], type='F_nu'):
    try:
        wv, fl, sig = spec  #[:, ~np.isnan(spec[1])]  # synthetic spectrum
    except:
        wv, fl = spec  #[:, ~np.isnan(spec[1])]
        sig = fl.copy() * .1
    if len(cov) == 0:
        cov = np.diag(sig**2)

    wv_flt, tra_flt = flt[:, 0], flt[:, 1]  # filter

    # if (wv_flt[9] < wv[0]) or (wv_flt[-10] > wv[-1]):
    #         return np.nan, np.nan
    tra_interp = np.interp(x=wv, xp=wv_flt, fp=tra_flt)
    tra_interp[(wv < wv_flt[0]) | (wv > wv_flt[-1])] = 0

    # effective bandwidth
    dlambda = np.diff(wv)
    width = (dlambda * tra_interp[1:] *
             wv[1:]).sum()  # for photon counters: weight ~ T/(h*nu) ~ T*lambda
    # convolution
    weight = tra_interp[1:] * wv[1:] / width * dlambda
    flux = (fl[1:] * weight).sum()  # erg cm-2 s-1 AA-1
    flux_err = (lax.dot(lax.dot(weight, cov[1:, 1:]),
                        weight.T))**.5  # erg cm-2 s-1 AA-1
    if type == 'F_lambda':
        return flux, flux_err
    # effective wavelength
    wv_eff = (wv[1:] * weight).sum()
    dlambda_dnu = wv_eff / (2.99792458e10 / (wv_eff * 1e-8))  # Ang/Hz
    flux_Jy = flux * dlambda_dnu * 1e23  # cgs to jansky
    flux_err_Jy = flux_err / flux * flux_Jy
    return flux_Jy, flux_err_Jy


def spec_to_mag(spec, flt, cov=[]):
    flux, flux_err = spec_to_flux(spec, flt, cov=cov)  # total flux in Jansky

    mag = -2.5 * np.log10(flux / 3631)
    mag_unc = -2.5 * np.log10((flux - flux_err) / (flux + flux_err)) / 2
    return mag, mag_unc