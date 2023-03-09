import numpy as np
import matplotlib.pyplot as plt
import glob

from tinygp import kernels, GaussianProcess, transforms
from tinygp.kernels.distance import L2Distance

import jax
import jax.numpy as jnp
import jaxopt

jax.config.update("jax_enable_x64", True)

import sys

sys.path.append('./tools/')
from data_binning import data_binning
from spec_to_flux import spec_to_mag, spec_to_flux

## 2D grid
## phase: -20 to 10 days, step = 1 day
## wavelength: 3750 to 7500 Ang, step = 50 Ang

phase_grid0 = np.arange(start=-20, stop=11)
wv_grid0 = np.arange(start=3750, stop=7525, step=25)
L_phase = len(phase_grid0)
L_wv = len(wv_grid0)
phase_grid, wv_grid = np.meshgrid(phase_grid0, wv_grid0)


# building the mean template
class prior_Ia:

    def __init__(self, X, fls, fls_unc, gp_params=None, time=False):
        '''
        generate a tinygp.GP object fitting the (logarithmic) spectral sequence of a template
        '''

        self.lgfls = np.log10(fls)
        self.lgfl_uncs = (np.log10(fls + fls_unc) -
                          np.log10(fls - fls_unc)) / 2

        if gp_params == None:
            print('Building a spectral template using Gaussian process...')
            self.gp = train_gp(2,
                               build_gp_func=build_gp,
                               X=X,
                               y=self.lgfls.ravel(),
                               yerr=self.lgfl_uncs.ravel())
        else:
            self.gp = build_gp(gp_params, X=X, yerr=self.lgfl_uncs.ravel())

#         if time:
#             import timeit
#             imports_and_vars=globals()
#             imports_and_vars.update(locals())
#             codes = '''
# X_new = np.array([phase_grid.ravel(), wv_grid.ravel()]).T
# self.lgfl_pred, self.cov_pred = self.gp.predict(self.lgfls.ravel(),
#                                                 X_new,
#                                                 return_cov=True)
# self.var_pred = np.diag(self.cov_pred)'''

#             N = 10
#             res = timeit.repeat(stmt=codes, globals=imports_and_vars, number=N, repeat=5)
#             print('{:.4f} +/- {:.4f}'.format(np.mean(res) / N, np.std(res, ddof=1) / N))

        X_new = np.array([phase_grid.ravel(), wv_grid.ravel()]).T
        self.lgfl_pred, self.cov_pred = self.gp.predict(self.lgfls.ravel(),
                                                        X_new,
                                                        return_cov=True)
        self.var_pred = np.diag(self.cov_pred)

    def get_spec(self, phase, get_cov=False, get_log=True):
        X_new = np.array([np.repeat(phase, L_wv), wv_grid0]).T
        lgfl_pred, cov_pred = self.gp.predict(self.lgfls.ravel(),
                                              X_new,
                                              return_cov=True)
        if not get_cov:
            unc_pred = np.diag(cov_pred)**.5
        else:
            unc_pred = cov_pred
        if get_log:
            return lgfl_pred, unc_pred
        lnfl_pred, lnfl_pred_cov = lgfl_pred * np.log(10), cov_pred * np.log(
            10)**2
        fl_pred = np.atleast_2d(np.exp(lnfl_pred + np.diag(lnfl_pred_cov)**.5))
        fl_cov_pred = fl_pred.T @ fl_pred * (np.exp(lnfl_pred_cov) - 1)
        if get_cov:
            return fl_pred.ravel(), fl_cov_pred
        else:
            return fl_pred.ravel(), np.diag(fl_cov_pred)**.5

    def get_syn_mag(self, phase, flt, z=0):
        fl_pred, fl_cov_pred = self.get_spec(phase,
                                             get_cov=True,
                                             get_log=False)
        mag, mag_unc = spec_to_mag(spec=[wv_grid0 * (1 + z), fl_pred / (1 + z)],
                                   cov=fl_cov_pred / (1 + z)**2,
                                   flt=flt)
        return mag, mag_unc

    def get_syn_flux(self, phase, flt, z=0):
        fl_pred, fl_cov_pred = self.get_spec(phase,
                                             get_cov=True,
                                             get_log=False)
        fl, fl_unc = spec_to_flux(spec=[wv_grid0 * (1 + z), fl_pred / (1 + z)],
                                  cov=fl_cov_pred / (1 + z)**2,
                                  flt=flt,
                                  type='F_lambda')
        return fl, fl_unc


# mean template using the spectral sequence of SN 2011fe
class prior_11fe(prior_Ia):

    def __init__(self, gp_params=None, time=False):
        files = glob.glob('./SN2011fe/Pereira_2013/*.dat')
        files.sort()
        phase = np.array([
            -15.2, -14.3, -13.3, -12.2, -11.3, -10.3, -9.3, -8.3, -7.2, -6.3,
            -5.3, -1.3, -0.3, 0.7, 1.7, 2.7, 3.7, 6.7, 8.7, 11.7, 13.7, 16.7,
            18.7, 21.7, 23.7, 74.1, 77.1, 79.1, 82.1, 87.1, 89.1, 97.1
        ])
        files = np.array(files)[phase <= 15]
        phase = phase[phase <= 15]
        wvs, fls, uncs = [], [], []

        # flux units = erg/s/cm2/A
        distmod = 29.04
        f_ratio = 10**(0.4 * distmod)

        for f in files:
            dat = np.loadtxt(f)
            dat = dat[(dat[:, 0] / (1 + 0.0008) > 3400) &
                      (dat[:, 0] / (1 + 0.0008) <
                       7700)]  # slightly wider range for flexibility
            wv, fl, unc = dat[:,
                              0], dat[:, 1] * f_ratio, dat[:, 2]**.5 * f_ratio
            wv_res = wv / (1 + 0.0008)
            bin = data_binning(np.array([wv_res, fl, unc]).T,
                               size=25)  # binsize = 25 A
            wvs.append(bin[:, 0])
            fls.append(bin[:, 1])
            uncs.append(bin[:, 2])

        wvs, fls, uncs = np.array(wvs), np.array(fls), np.array(uncs)

        X = np.array([np.repeat(phase, len(wvs[0])), wvs.ravel()]).T
        super().__init__(X=X,
                         fls=fls,
                         fls_unc=uncs,
                         gp_params=gp_params,
                         time=time)


# Eric Hsiao's template
class prior_Hsiao(prior_Ia):

    def __init__(self, gp_params=None, time=False):
        dat = np.loadtxt('./hsiao_template/snflux_1a.dat')

        phases0, wvs0, fls0 = dat[:, 0].reshape(106, 2401), dat[:, 1].reshape(
            106, 2401), dat[:, 2].reshape(106, 2401)
        L_phase_Hsiao = ((phases0[:, 0] >= -18) & (phases0[:, 0] <= 12)).sum()
        L_wv_Hsiao = ((wvs0[0] > 3400) & (wvs0[0] < 7700)).sum()
        cut = (
            (phases0 >= -18)) & (phases0 <= 12) & (wvs0 > 3400) & (wvs0 < 7700)
        wvs = wvs0[cut].reshape(L_phase_Hsiao, L_wv_Hsiao)
        fls = fls0[cut].reshape(L_phase_Hsiao, L_wv_Hsiao) * 10**(0.4 * 19.5)
        phases = phases0[cut].reshape(L_phase_Hsiao, L_wv_Hsiao)
        uncs = fls * 1e-2

        wvs_bin, fls_bin, uncs_bin = [], [], []
        for k in range(L_phase_Hsiao):
            bin = data_binning(np.array([wvs[k], fls[k], uncs[k]]).T,
                               size=50)  # binsize = 50 A
            wvs_bin.append(bin[:, 0])
            fls_bin.append(bin[:, 1])
            uncs_bin.append(bin[:, 2])

        wvs_bin, fls_bin, uncs_bin = np.array(wvs_bin), np.array(
            fls_bin), np.array(uncs_bin)

        X = np.array([np.repeat(phases[:, 0], len(wvs_bin[0])), wvs_bin.ravel()]).T

        super().__init__(X=X,
                         fls=fls_bin,
                         fls_unc=uncs_bin,
                         gp_params=gp_params,
                         time=time)


def train_gp(nparams, build_gp_func, X, y, yerr):

    @jax.jit
    def loss(params):
        return -build_gp_func(params, X, yerr).log_probability(y)

    params = {
        "log_scale": np.log([18.6, 726])[:nparams],
        "mean": np.float64(-2.8),
    }
    print('Initial log_p = {:.1f}'.format(-loss(params)))
    solver = jaxopt.ScipyMinimize(fun=loss)
    soln = solver.run(params)
    print(np.exp(soln.params["log_scale"]), soln.params["mean"])
    print('Final log_p = {:.1f}'.format(-loss(soln.params)))
    return build_gp_func(soln.params, X, yerr)


def build_gp(params, X, yerr):
    kernel = transforms.Linear(
        jnp.exp(-params["log_scale"]), kernels.Matern32(
            distance=L2Distance()))  # default : L1Distance - causing nan's
    return GaussianProcess(kernel, X, diag=yerr**2, mean=params["mean"])