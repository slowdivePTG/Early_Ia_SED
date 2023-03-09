import numpy as np
import glob
import pandas as pd

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import Distance

import sys

from SN_phot_spec import SN_phot_spec

sys.path.append('./tools/')
from dust_extinction import calALambda
from data_binning import data_binning, plot_box_spec
from spec_to_flux import spec_to_mag

## 2D grid
## phase: -20 to 10 days, step = 1 day
## wavelength: 3750 to 7500 Ang, step = 25 Ang

phase_grid0 = np.arange(start=-20, stop=11)
wv_grid0 = np.arange(start=3750, stop=7525, step=25)
RV_grid0 = np.arange(start=-0.020, stop=0.055, step=0.005)
L_phase = len(phase_grid0)
L_wv = len(wv_grid0)
L_RV = len(RV_grid0)
phase_grid, wv_grid = np.meshgrid(phase_grid0, wv_grid0)


def CSP_SN(obj, prior, save=None, plot_syn_phot=True):

    csp_lc = pd.read_csv('./CSP_Photometry_DR3/SN_photo.dat')
    csp_lc['MJD'] = csp_lc['MJD'] + 53000 - 0.5
    obj_lc = csp_lc[csp_lc['Object'] == obj]
    lc = {}
    for flt in 'gr':
        lc[flt] = obj_lc[obj_lc['flt'] == flt]

    csp_spec_meta = pd.read_csv('./CSP_spectra_DR1/table1.dat')
    obj_spec_meta = csp_spec_meta[csp_spec_meta['SN'] == obj[2:]]
    spec_meta = {}
    spec_meta['z'] = float(obj_spec_meta['z_Helio'])
    z_CMB = float(obj_spec_meta['z_CMB'])
    cosmo = FlatLambdaCDM(H0=73, Om0=0.3)
    dis = Distance(z=z_CMB, cosmology=cosmo, unit=u.pc).value
    spec_meta['distmod'] = 5 * np.log10(dis / 10)
    spec_meta['T_max'] = float(obj_spec_meta['T_B(max)']) - 0.5
    spec_meta['EBV'] = float(obj_spec_meta['E(B - V)'])

    spec_files = glob.glob('./CSP_spectra_DR1/*{}*'.format(obj[4:]))
    spec_files.sort()

    flts = glob.glob('./CSP_Photometry_DR2/*tel_ccd*1.2.dat')
    flts.sort()
    flts = [flts[0], flts[2]]

    return SN_phot_spec(obj_lc=lc,
                        obj_spec_meta=spec_meta,
                        obj_spec_files=spec_files,
                        flts=flts,
                        prior=prior,
                        plot_syn_phot=plot_syn_phot)
