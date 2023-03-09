import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import pymc as pm
import arviz as az
import george
from george import kernels
import xarray as xr

from scipy.interpolate import interp1d

import sys

sys.path.append('./tools/')
from dust_extinction import calALambda
from data_binning import data_binning, plot_box_spec
from spec_to_flux import spec_to_mag

## 2D grid
## phase: -20 to 10 days, step = 1 day
## wavelength: 3700 to 7500 Ang, step = 25 Ang
## mock synthetic photometry: RV/c = -0.02 to 0.05, step = 0.005

phase_grid0 = np.arange(start=-20, stop=11)
wv_grid0 = np.arange(start=3700, stop=7525, step=25)
RV_grid0 = np.arange(start=-0.020, stop=0.055, step=0.005)
L_phase = len(phase_grid0)
L_wv = len(wv_grid0)
L_RV = len(RV_grid0)
phase_grid, wv_grid = np.meshgrid(phase_grid0, wv_grid0)


class SN_phot_spec:

    def __init__(self,
                 obj_lc,
                 obj_spec_meta,
                 obj_spec_files,
                 flts,
                 prior,
                 save=None,
                 plot_syn_phot=True):
        '''
        Parameters
        ----------
        obj_lc : dict
            multiband light curves
            each element is a dataframe
            columns - MJD, mag, mag_err
        obj_spec_meta : dict
            z - heliocentric redshift
            distmod - distance modulus (mag)
            T_max - epoch of maximum light (in B by default)
            EBV - E(B - V)
        object_spec_files : list
            a list of spec files
        flts : list
            data files of throughput functions (2D arrays)
        prior : pymc.gp object
            the spectral model of 11fe as a prior
        '''

        self.prior = prior
        self.obj_lc = obj_lc
        self.obj_spec_meta = obj_spec_meta
        self.flts = flts
        print(self.obj_spec_meta)

        self.z = float(self.obj_spec_meta['z'])
        self.distmod = float(self.obj_spec_meta['distmod'])
        dis = 10**(0.4 * self.distmod / 2) * 10  # pc
        self.MJD_max = float(self.obj_spec_meta['T_max'])
        self.EBV = float(self.obj_spec_meta['EBV'])

        self.obj_spec = obj_spec_files
        self.obj_spec.sort()

        cols = ['green', 'orange']

        gp_lc = []
        wv_eff = []

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for j in range(2):
            # read light curves
            lc = self.obj_lc['gr'[j]]
            phase_lc = (lc['MJD'] - self.MJD_max) / (1 + self.z)
            ax[0].errorbar(phase_lc,
                           lc['mag'] - self.distmod,
                           yerr=lc['mag_err'],
                           fmt='o',
                           color=cols[j],
                           label='gr'[j])

            # interpolation with GP
            kernel_lc = kernels.Matern32Kernel(20**2)
            gp_temp = george.GP(kernel=kernel_lc,
                                mean=np.mean(lc['mag'].values),
                                fit_mean=True)
            gp_temp.compute(phase_lc.values.reshape(-1, 1),
                            lc['mag_err'].to_numpy())
            # print('Fitting {}-band light curve'.format('gr'[j]))
            # res = optimize_gp(gp=gp_temp, y=lc['mag'].to_numpy())
            # print(f'l_t = {np.exp(res[1]/2)}')
            # gp_temp.set_parameter_vector(res)
            gp_lc.append(gp_temp)
            t_grid = np.linspace(-20, 10, 100)
            mag_grid, mag_var_grid = gp_temp.predict(
                self.obj_lc['gr'[j]]['mag'].to_numpy(),
                t_grid.T,
                return_var=True)
            ax[0].fill_between(t_grid,
                               mag_grid - self.distmod - mag_var_grid**.5,
                               mag_grid - self.distmod + mag_var_grid**.5,
                               color=cols[j],
                               alpha=.3)
            # calculate effective wavelength given the throughput function
            dat = np.loadtxt(self.flts[j])
            wv_eff = np.append(wv_eff,
                               (dat[:, 0] * dat[:, 1]).sum() / dat[:, 1].sum())

        self.gp_lc = gp_lc
        self.wv_eff = wv_eff
        ax[0].invert_yaxis()
        ax[0].legend(prop={'size': 20})

        phase_spec = []
        lgfl_obj, unc_obj = [], []
        self.delta_mag_z = {'g': [], 'r': []}
        self.delta_mag_err_z = {'g': [], 'r': []}
        for k, spec in enumerate(self.obj_spec):
            # read spectra
            with open(spec) as f:
                lines = f.readlines()
                for l in lines:
                    if 'Epoch' in l.split()[0]:
                        phase = float(l.split()[-1])
                if phase < phase_lc.values[0] - 0.5:
                    continue  # avoid early spectra without photometry
                if phase > 10:
                    break
            raw = np.loadtxt(spec)
            spec_ori = raw.copy()
            spec_ori[:, 0] = raw[:, 0] * (1 + self.z)

            # mangling
            Dmag = np.empty(2)
            for j in range(2):
                lc = self.obj_lc['gr'[j]]
                phase_lc = (lc['MJD'] - self.MJD_max) / (1 + self.z)

                # original synthetic photometry
                mag_ori, _ = spec_to_mag(spec_ori.T,
                                         flt=np.loadtxt(self.flts[j]))
                if plot_syn_phot:
                    ax[0].scatter(phase,
                                  mag_ori - self.distmod,
                                  color='white',
                                  marker='*',
                                  edgecolors=cols[j])
                mag_fit, mag_fit_var = gp_lc[j].predict(lc['mag'].to_numpy(),
                                                        phase,
                                                        return_var=True)

                Dmag[j] = mag_ori - mag_fit[0]

            spec_cal = np.ones_like(raw)
            spec_cal[:, 0] = raw[:, 0] * (1 + self.z)
            spec_cal[:, 1] = raw[:, 1] * 10**(0.4 * interp1d(
                x=wv_eff, y=Dmag, fill_value='extrapolate')(spec_cal[:, 0]))

            for j in range(2):
                if len(wv_eff[~np.isnan(Dmag)]) <= 1:
                    break
                # synthetic photometry after mangling
                mag, _ = spec_to_mag(spec_cal.T, flt=np.loadtxt(self.flts[j]))
                if plot_syn_phot:
                    ax[0].scatter(phase,
                                  mag - self.distmod,
                                  color='white',
                                  marker='s',
                                  edgecolors=cols[j])
                mag_fit, mag_fit_var = gp_lc[j].predict(
                    self.obj_lc['gr'[j]]['mag'].to_numpy(),
                    phase,
                    return_var=True)
                # print(
                #     phase, 'gr'[j], 'synthetic - GP fit = {:.4f}'.format(
                #         mag - self.distmod -
                #         calALambda(self.wv_eff[j], EBV=self.EBV, RV=3.1)))

            if len(wv_eff[~np.isnan(Dmag)]) <= 1:
                continue
            phase_spec.append(phase)
            binned = data_binning(np.array(
                [raw[:, 0], spec_cal[:, 1],
                 np.ones_like(spec_cal[:, 1])]).T,
                                  size=25)
            fl_interp = np.interp(x=wv_grid0, xp=binned[:, 0], fp=binned[:, 1])
            #     unc_interp = np.interp(
            #         x=wv_grid0, xp=binned[:, 0],
            #         fp=binned[:, 2]**2)**.5  #TODO: avoid interpolating uncertainty
            Fl_interp = fl_interp * (dis / 10)**2 * 10**(
                0.4 * calALambda(wv_grid0 *
                                 (1 + self.z), EBV=self.EBV, RV=3.1))
            unc_interp = np.min(Fl_interp) * np.ones_like(
                Fl_interp) * 0.1  # assuming SNR >= 10 after binning

            lgfl_11fe, lgfl_11fe_unc = prior.get_spec(phase)
            lgfl_obj.append(np.log10(Fl_interp))
            unc_obj.append((np.log10(
                (Fl_interp + unc_interp) / (Fl_interp - unc_interp))) / 2)

            ax[1].plot(wv_grid0,
                       np.log10(Fl_interp) - lgfl_11fe.mean() -
                       len(phase_spec),
                       color='k')
            ax[1].plot(wv_grid0,
                       lgfl_11fe - lgfl_11fe.mean() - len(phase_spec),
                       color='0.8')
            ax[1].text(7500,
                       lgfl_11fe[-1] - lgfl_11fe.mean() - len(phase_spec),
                       phase,
                       fontsize=18)
            # ax[2].plot(wv_grid0,
            #            np.log10(Fl_interp),
            #            color=plt.cm.RdBu(
            #                (phase - phase_spec[0]) / (10 - phase_spec[0])))

            # mags from spectra
            delta_mag_z = {'g': [], 'r': []}
            delta_mag_err_z = {'g': [], 'r': []}
            fl_11fe, fl_cov_11fe = self.prior.get_spec(phase=phase,
                                                       get_cov=True,
                                                       get_log=False)
            for j in range(2):
                for l, RV in enumerate(RV_grid0):
                    mag, mag_err = spec_to_mag(spec=[
                        wv_grid0 * (1 + RV),
                        Fl_interp,
                        unc_interp,
                    ],
                                               flt=np.loadtxt(flts[j]))
                    mag_11fe, mag_err_11fe = spec_to_mag(
                        spec=[wv_grid0 * (1 + RV),
                              fl_11fe.ravel()],
                        cov=fl_cov_11fe,
                        flt=np.loadtxt(flts[j]))
                    delta_mag_z['gr'[j]].append(mag - mag_11fe)
                    delta_mag_err_z['gr'[j]].append(
                        (mag_err**2 + mag_err_11fe**2)**.5)
                    ax[2].errorbar(wv_eff[j] / (1 + RV),
                                   mag - len(phase_spec),
                                   yerr=mag_err,
                                   fmt='o',
                                   color='k',
                                   ms=2)
                    ax[2].errorbar(wv_eff[j] / (1 + RV),
                                   mag_11fe - len(phase_spec),
                                   fmt='o',
                                   color='0.8',
                                   ms=2)
                if j == 1:
                    ax[2].text(6900,
                               mag_11fe - len(phase_spec),
                               phase,
                               fontsize=18)
            self.delta_mag_z['g'].append(delta_mag_z['g'])
            self.delta_mag_err_z['g'].append(delta_mag_err_z['g'])
            self.delta_mag_z['r'].append(delta_mag_z['r'])
            self.delta_mag_err_z['r'].append(delta_mag_err_z['r'])

        # mags from light curves
        self.lc_mag = {'g': [], 'r': []}
        self.lc_mag_err = {'g': [], 'r': []}
        self.lc_delta_mag = {'g': [], 'r': []}
        self.lc_delta_mag_err = {'g': [], 'r': []}
        self.phase_lc = {}
        for j in range(2):
            lc = self.obj_lc['gr'[j]]
            phase_lc = (lc['MJD'].to_numpy() - self.MJD_max) / (1 + self.z)
            mag, mag_unc = lc['mag'].to_numpy(), lc['mag_err'].to_numpy()
            mag = mag - self.distmod - calALambda(
                self.wv_eff[j], EBV=self.EBV, RV=3.1)
            phase_index = phase_lc <= np.array(phase_spec).max()
            mag, mag_unc = mag[phase_index], mag_unc[phase_index]
            phase_lc = phase_lc[phase_index]
            self.phase_lc['gr'[j]] = phase_lc

            for k, phase in enumerate(phase_lc):
                mag_11fe, mag_err_11fe = self.prior.get_mag(phase=phase,
                                                            flt=np.loadtxt(
                                                                self.flts[j]),
                                                            z=self.z)
                self.lc_mag['gr'[j]].append(mag[k])
                self.lc_mag_err['gr'[j]].append(mag_unc[k])
                self.lc_delta_mag['gr'[j]].append(mag[k] - mag_11fe)
                self.lc_delta_mag_err['gr'[j]].append(
                    (mag_unc[k]**2 + mag_err_11fe**2)**.5)

        ax[0].set_yticks([-16, -16.5, -17, -17.5, -18, -18.5, -19, -19.5])
        ax[0].set_xticks([-20, -10, 0, 10])
        ax[1].set_yticks([])
        ax[2].set_yticks([])
        ax[2].invert_yaxis()
        ax[0].set_xlabel(r'$t-t_{\mathrm{max},B}\ [\mathrm{day}]$')
        ax[0].set_ylabel(r'$M$')
        ax[1].set_ylabel(r'$\log F_\lambda+\mathrm{offset}$')
        ax[2].set_ylabel(r'$M+\mathrm{offset}$')
        ax[1].set_xlabel(r'$\lambda$')
        ax[2].set_xlabel(r'$\lambda_\mathrm{eff}$')

        ax[0].set_xlim(-22, 11)
        ax[0].set_ylim(-15.8, -19.6)
        ax[1].set_xlim(3300, 8400)
        ax[2].set_xlim(4300, 7400)
        plt.tight_layout()
        if save != None:
            plt.savefig(save, bbox_inches='tight')
        plt.show()

        self.phase_spec = np.array(phase_spec)
        self.lgfl_obj = np.array(lgfl_obj)
        self.unc_obj = np.array(unc_obj)

    # def MCMC_syn_phot_RV(self,
    #                      index_train,
    #                      index_validate,
    #                      l_t=None,
    #                      l_z=None,
    #                      A=None,
    #                      plot=False,
    #                      optimize=False,
    #                      find_MAP=False):
    #     '''
    #     - 11fe templates as prior
        
    #     Parameters
    #     -------------------
    #     index_train: array-like
    #         Phase indices of the spectra used in the GP fitting
    #     '''

    #     # mags from spectra
    #     spec_delta_mag = self.delta_mag_z
    #     spec_delta_mag_err = self.delta_mag_err_z
    #     lc_delta_mag = self.lc_delta_mag
    #     lc_delta_mag_err = self.lc_delta_mag_err
    #     lc_phase = self.phase_lc

    #     delta_gmag = np.append(
    #         np.array(spec_delta_mag['g'])[index_train].ravel(),
    #         lc_delta_mag['g'])
    #     delta_gmag_err = np.append(
    #         np.array(spec_delta_mag_err['g'])[index_train].ravel(),
    #         lc_delta_mag_err['g'])
    #     z_grid_g = np.append(
    #         np.repeat([RV_grid0], len(self.phase_spec[index_train]),
    #                   axis=0).ravel(), [self.z] * len(lc_phase['g']))
    #     phase_grid_g = np.append(np.repeat(self.phase_spec[index_train], L_RV),
    #                              lc_phase['g'])
    #     delta_rmag = np.append(
    #         np.array(spec_delta_mag['r'])[index_train].ravel(),
    #         lc_delta_mag['r'])
    #     delta_rmag_err = np.append(
    #         np.array(spec_delta_mag_err['r'])[index_train].ravel(),
    #         lc_delta_mag_err['r'])
    #     z_grid_r = np.append(
    #         np.repeat([RV_grid0], len(self.phase_spec[index_train]),
    #                   axis=0).ravel(), [self.z] * len(lc_phase['r']))
    #     phase_grid_r = np.append(np.repeat(self.phase_spec[index_train], L_RV),
    #                              lc_phase['r'])
    #     if plot:
    #         fig, ax = plt.subplots(2,
    #                                1,
    #                                figsize=(7, 9),
    #                                sharex=True,
    #                                sharey=True)
    #         vmin = np.append(delta_gmag, delta_rmag).min()
    #         vmax = np.append(delta_gmag, delta_rmag).max()
    #         ax[0].scatter(phase_grid_g,
    #                       z_grid_g,
    #                       c=delta_gmag,
    #                       cmap=plt.cm.turbo,
    #                       vmin=vmin,
    #                       vmax=vmax)
    #         im = ax[1].scatter(phase_grid_r,
    #                            z_grid_r,
    #                            c=delta_rmag,
    #                            cmap=plt.cm.turbo,
    #                            vmin=vmin,
    #                            vmax=vmax)
    #         ax[1].set_xlabel('phase')
    #         ax[0].set_ylabel('RV/c')
    #         ax[1].set_ylabel('RV/c')
    #         ax[1].xaxis.set_minor_locator(MultipleLocator(1))
    #         ax[1].xaxis.set_major_locator(MultipleLocator(5))
    #         ax[1].yaxis.set_minor_locator(MultipleLocator(.01))

    #         fig.subplots_adjust(right=0.8)
    #         cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #         cbar_ax.xaxis.set_minor_locator(MultipleLocator(.05))
    #         cbar_ax.set_xlabel('Dmag')
    #         fig.colorbar(im, cax=cbar_ax)
    #         plt.show()

    #     X_g = np.array([phase_grid_g, z_grid_g]).T
    #     X_r = np.array([phase_grid_r, z_grid_r]).T
    #     Xnew = np.atleast_2d(
    #         [np.repeat(self.phase_spec[index_validate][0], L_RV), RV_grid0]).T

    #     kernel_lc = A * kernels.Matern32Kernel([l_t**2, l_z**2], ndim=2)
    #     gp_g = george.GP(kernel=kernel_lc, mean=0, fit_mean=False)
    #     gp_r = george.GP(kernel=kernel_lc, mean=0, fit_mean=False)
    #     gp_g.compute(X_g, delta_gmag_err)
    #     gp_r.compute(X_r, delta_rmag_err)
    #     delta_gmag_pred, var_gmag_pred = gp_g.predict(delta_gmag, Xnew, return_var=True)
    #     delta_rmag_pred, var_rmag_pred = gp_r.predict(delta_rmag, Xnew, return_var=True)

    #     fl_11fe, fl_cov_11fe = self.prior.get_spec(
    #             self.phase_spec[index_validate][0],
    #             get_cov=True,
    #             get_log=False)
    #     gmag_11fe, rmag_11fe = [], []
    #     gmag_err_11fe, rmag_err_11fe = [], []
    #     for RV in RV_grid0:
    #         gmag, gmag_err = spec_to_mag(spec=[wv_grid0 * (1 + RV), fl_11fe],
    #                                 cov=fl_cov_11fe,
    #                                 flt=np.loadtxt(self.flts[0]))
    #         rmag, rmag_err = spec_to_mag(spec=[wv_grid0 * (1 + RV), fl_11fe],
    #                                 cov=fl_cov_11fe,
    #                                 flt=np.loadtxt(self.flts[1]))
    #         gmag_11fe = np.append(gmag_11fe, gmag)
    #         rmag_11fe = np.append(rmag_11fe, rmag)
    #         gmag_err_11fe = np.append(gmag_err_11fe, gmag_err)
    #         rmag_err_11fe = np.append(rmag_err_11fe, rmag_err)

    #     gmag_pred = gmag_11fe + delta_gmag_pred
    #     gmag_unc_pred = (var_gmag_pred + gmag_err_11fe**2)**.5
    #     rmag_pred = rmag_11fe + delta_rmag_pred
    #     rmag_unc_pred = (var_rmag_pred + rmag_err_11fe**2)**.5

    #     if optimize:
    #         vector = optimize_gp([gp_g, gp_r], [delta_gmag, delta_rmag])
    #         print(np.exp(vector[0]), np.exp(vector[1]/2), np.exp(vector[2]/2))
    #         gp_g.set_parameter_vector(vector)
    #         gp_r.set_parameter_vector(vector)
    #         delta_gmag_pred, var_gmag_pred = gp_g.predict(delta_gmag, Xnew, return_var=True)
    #         delta_rmag_pred, var_rmag_pred = gp_r.predict(delta_rmag, Xnew, return_var=True)

    #     return (gmag_pred, gmag_unc_pred, rmag_pred, rmag_unc_pred)

        # with pm.Model() as Spec_phot:
        #     if l_t == None:
        #         lg_l_t = pm.Normal('lg_l_t', np.log10(20), 0.05)
        #         l_t = pm.Deterministic('l_t', 10**lg_l_t)
        #     if l_z == None:
        #         lg_l_z = pm.Normal('lg_l_z', np.log10(0.02), 0.1)
        #         l_z = pm.Deterministic('l_z', 10**lg_l_z)
        #     if A == None:
        #         lg_A = pm.Normal('lg_A', -1, 0.01)
        #         A = pm.Deterministic('A', 10**lg_A)

        #     # setting up the GP model
        #     cov = A * pm.gp.cov.Matern32(input_dim=2, ls=[l_t, l_z])
        #     gp_g = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov)
        #     gp_r = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov)

        #     Delta_mag_z_g = gp_g.marginal_likelihood('Delta_mag_z_g',
        #                                              X=np.atleast_2d(X_g),
        #                                              y=delta_gmag,
        #                                              noise=delta_gmag_err)

        #     Delta_mag_z_r = gp_r.marginal_likelihood('Delta_mag_z_r',
        #                                              X=np.atleast_2d(X_r),
        #                                              y=delta_rmag,
        #                                              noise=delta_rmag_err)

        #     Delta_mag_g = gp_g.conditional('Delta_mag_g', Xnew=Xnew)
        #     Delta_mag_r = gp_r.conditional('Delta_mag_r', Xnew=Xnew)
            
        #     mag_g = pm.Deterministic('mag_g', gmag_11fe + Delta_mag_g)
        #     mag_r = pm.Deterministic('mag_r', rmag_11fe + Delta_mag_r)

        # with Spec_phot:
        #     if find_MAP:
        #         res = pm.find_MAP()
        #         return res
        #     else:
        #         trace = pm.sample(chains=2,
        #                           tune=500,
        #                           draws=500,
        #                           return_inferencedata=True,
        #                           target_accept=0.95)
        #     return trace

    def MCMC_spec_lc_fixed_kernel(self, index_train, l_t=None):
        '''
        sampling the observed light curves & spectra
        - 11fe templates as prior
        - fixed kernel parameters
        
        Parameters
        -------------------
        index_train: array-like
            Phase indices of the spectra used in the GP fitting
        '''

        Alambda = calALambda(wv_grid0 * (1 + self.z), EBV=self.EBV, RV=3.1)

        wv_test, phase_test = np.meshgrid(wv_grid0,
                                          self.phase_spec[index_train])
        X = np.array([phase_test.ravel(), wv_test.ravel()]).T

        with pm.Model() as Spec_phot:

            # MAP model for 11fe
            with self.prior.gp_model:
                lg_f_11fe_mean, lg_f_11fe_cov = self.prior.gp.predict(
                    Xnew=np.atleast_2d(X), point=self.prior.gp_res)

            if l_t == None:
                l_t = float(self.prior.gp_res['l_t'])
            l_w = float(self.prior.gp_res['l_wv'])
            A = 1  #float(self.prior.gp_res['A'])

            # setting up the GP model
            cov = A * pm.gp.cov.Matern32(input_dim=2, ls=[l_t, l_w])
            gp = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov)

            # GP spectra v.s. observed spectra
            lg_Delta_F0 = gp.marginal_likelihood(
                'lg_Delta_F0',
                X=np.atleast_2d(X),
                y=self.lgfl_obj[index_train].ravel() - lg_f_11fe_mean,
                noise=self.unc_obj[index_train].ravel())

            # synthetic photometry v.s. light curve
            for j in range(2):
                lc = self.obj_lc['gr'[j]]
                phase_lc = (lc['MJD'].to_numpy() - self.MJD_max) / (1 + self.z)
                mag, mag_unc = lc['mag'].to_numpy(), lc['mag_err'].to_numpy()
                phase_index = (phase_lc <= self.phase_spec.max()) & (
                    phase_lc >= self.phase_spec.min() - 7
                )  # only fit the photometry sufficiently close to the spectra
                mag, mag_unc = mag[phase_index], mag_unc[phase_index]
                phase_lc = phase_lc[phase_index]
                for k in range(len(mag)):
                    Xnew = np.array([np.repeat(phase_lc[k], L_wv), wv_grid0]).T
                    lg_Delta_F = gp.conditional('lg_Delta_F_{}_{:.0f}'.format(
                        'gri'[j], k),
                                                Xnew=np.atleast_2d(Xnew))
                    with self.prior.gp_model:
                        lg_f_11fe_mean_temp, lg_f_11fe_cov_temp = self.prior.gp.predict(
                            Xnew=np.atleast_2d(Xnew), point=self.prior.gp_res)
                    lg_Flambda = pm.Deterministic(
                        'lg_F_lambda_{}_{:.0f}'.format('gri'[j], k),
                        lg_Delta_F + lg_f_11fe_mean_temp)
                    Mag_mean = pm.Deterministic(
                        'Mag_mean_{}_{:.0f}'.format('gri'[j], k),
                        spec_to_mag(spec=[
                            wv_grid0 * (1 + self.z),
                            10**(lg_Flambda - 0.4 * Alambda),
                            np.ones(L_wv)
                        ],
                                    flt=np.loadtxt(self.flts[j]))[0])
                    Mag = pm.Normal('Mag_{}_{:.0f}'.format('gri'[j], k),
                                    Mag_mean,
                                    mag_unc[k],
                                    observed=mag[k] - self.distmod)
        with Spec_phot:
            res = pm.find_MAP()
            # for j in range(2):
            #     print('gr'[j])
            #     keys = [key for key in res.keys() if 'Mag_mean_{}'.format('gr'[j]) in key]
            #     keys.sort()
            #     for k, key in enumerate(keys):
            #         print(self.obj_lc['gr'[j]]['mag'].to_numpy()[k] - self.distmod, '+/-', self.obj_lc['gr'[0]]['mag_err'].to_numpy()[k])
            #         print(res[key])
            trace = pm.sample(chains=2,
                              tune=1500,
                              draws=500,
                              return_inferencedata=True,
                              target_accept=0.95)
        return res, trace

    def MCMC_spec_lc_grid(self, index_train, l_t_grid=np.arange(20) * 10 + 10):
        '''
        sampling the observed light curves & spectra
        - 11fe templates as prior
        - a grid of l_t
        '''

        ln_like_spec = np.empty_like(l_t_grid, dtype=float)
        ln_like_mag = np.empty_like(l_t_grid, dtype=float)
        #ln_p = np.empty_like(l_t_grid, dtype=float)
        dic = {}
        for k, l_t in enumerate(l_t_grid):
            print(f'l_t = {l_t}')
            _, trace = self.MCMC_spec_lc_fixed_kernel(index_train, l_t=l_t)
            mag_keys = [
                key for key in trace.log_likelihood.keys() if 'Mag' in key
            ]
            ln_like_mag_temp = 0
            for key in mag_keys:
                ln_like_mag_temp = ln_like_mag_temp + trace.log_likelihood[
                    key].to_numpy()
            ln_like_spec_temp = trace.log_likelihood.lg_Delta_F0.to_numpy()
            trace.log_likelihood["sum"] = xr.DataArray(
                ln_like_spec_temp + ln_like_mag_temp,
                dims=("chain", "draw", "sum_dim"))
            print(az.loo(trace, var_name='sum'))
            dic[f'l_t = {l_t}'] = trace

            ln_like_spec[k] = np.mean(ln_like_spec_temp.ravel())
            ln_like_mag[k] = np.mean(ln_like_mag_temp.ravel())
            #ln_p[k] = np.max(trace.sample_stats.lp.to_numpy().ravel()) log_p up to a constant (evidence)

        ln_like_spec -= ln_like_spec.max()  # remove a constant
        ln_like_mag -= ln_like_mag.max()  # remove a constant
        df_comparison = az.compare(dic, var_name='sum')
        return ln_like_spec, ln_like_mag, df_comparison

    def MCMC_spec_phot_fixed_kernel(self,
                                    index_train,
                                    index_validate,
                                    l_t=None,
                                    use_observation=True,
                                    nchains=2,
                                    ndraws=500,
                                    nburns=1000):
        '''
        sampling the spectrum at a given phase
        - 11fe templates as prior
        - fixed kernel parameters
        '''

        Alambda = calALambda(wv_grid0 * (1 + self.z), EBV=self.EBV, RV=3.1)

        wv_test, phase_test = np.meshgrid(wv_grid0,
                                          self.phase_spec[index_train])
        X = np.array([phase_test.ravel(), wv_test.ravel()]).T

        if l_t == None:
            with pm.Model() as Phot:
                print('Searching for MAP estimators for l_t and l_wv...')
                ln_l_t = pm.Uniform('ln_l_t', np.log(10), np.log(500))
                l_t = pm.Deterministic('l_t', np.exp(ln_l_t))
                l_w = float(self.prior.gp_res['l_wv'])
                A = 1  #float(self.prior.gp_res['A'])

                # setting up the GP model
                cov = A * pm.gp.cov.Matern32(input_dim=2, ls=[l_t, l_w])
                gp = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov)

                # GP spectra v.s. observed spectra
                lg_f0 = gp.marginal_likelihood(
                    'lg_f0',
                    X=np.atleast_2d(X),
                    y=self.lgfl_obj[index_train].ravel(),
                    noise=self.unc_obj[index_train].ravel())
                res_2D_GP = pm.find_MAP()
                print(float(res_2D_GP['l_t']),
                      float(self.prior.gp_res['l_wv']))
            l_t = float(res_2D_GP['l_t'])

        with pm.Model() as Spec_phot:

            # MAP model for 11fe
            with self.prior.gp_model:
                lg_f_11fe_mean, lg_f_11fe_cov = self.prior.gp.predict(
                    Xnew=np.atleast_2d(X), point=self.prior.gp_res)

            # setting up the GP model
            cov = pm.gp.cov.Matern32(
                input_dim=2, ls=[l_t, float(self.prior.gp_res['l_wv'])])
            gp = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov)

            # GP spectra v.s. observed spectra
            lg_Delta_f0 = gp.marginal_likelihood(
                'lg_Delta_F0',
                X=np.atleast_2d(X),
                y=self.lgfl_obj[index_train].ravel() - lg_f_11fe_mean,
                noise=self.unc_obj[index_train].ravel())

            # synthetic photometry v.s. light curve
            Xnew = np.array(
                [np.repeat(self.phase_spec[index_validate], L_wv), wv_grid0]).T
            lg_Delta_f = gp.conditional('lg_Delta_F', Xnew=np.atleast_2d(Xnew))
            with self.prior.gp_model:
                lg_f_11fe_mean_temp, lg_f_11fe_cov_temp = self.prior.gp.predict(
                    Xnew=np.atleast_2d(Xnew), point=self.prior.gp_res)
            lg_Flambda = pm.Deterministic('lg_F_lambda',
                                          lg_Delta_f + lg_f_11fe_mean_temp)
            Mag_mean = []
            mag_mean_obs, mag_unc_obs = np.empty(2, dtype=float), np.empty(
                2, dtype=float)
            for j in range(2):
                lc = self.obj_lc['gr'[j]]
                phase_lc = (lc['MJD'].to_numpy() - self.MJD_max) / (1 + self.z)
                mag_mean_obs[j], mag_var_obs = self.gp_lc[j].predict(
                    lc['mag'].to_numpy(),
                    self.phase_spec[index_validate],
                    return_var=True)
                mag_unc_obs[j] = mag_var_obs**.5

                Mag_mean.append(
                    pm.Deterministic(
                        'Mag_mean_{}'.format('gri'[j]),
                        spec_to_mag(spec=[
                            wv_grid0 * (1 + self.z),
                            10**(lg_Flambda - 0.4 * Alambda),
                            np.ones(L_wv)
                        ],
                                    flt=np.loadtxt(self.flts[j]))[0]))
            if use_observation:
                Mag = pm.Normal('Mag',
                                Mag_mean,
                                mag_unc_obs,
                                observed=mag_mean_obs - self.distmod)
            K_g = pm.Deterministic(
                'K(g)',
                self.K_corr(wv=wv_grid0,
                            Flambda=10**lg_Flambda,
                            flt=np.loadtxt(self.flts[0])))
            K_r = pm.Deterministic(
                'K(r)',
                self.K_corr(wv=wv_grid0,
                            Flambda=10**lg_Flambda,
                            flt=np.loadtxt(self.flts[1])))
            K_g_r = pm.Deterministic('K(g-r)', K_g - K_r)

        with Spec_phot:
            res = pm.find_MAP()
            trace = pm.sample(chains=nchains,
                              tune=nburns,
                              draws=ndraws,
                              return_inferencedata=True)
        return res, trace

    def MCMC_spec_phot_grid(self,
                            index_train,
                            index_validate,
                            nchains=2,
                            ndraws=500,
                            nburns=1000,
                            l_t_grid=np.arange(20) * 10 + 10):
        '''
        sampling the observed light curves & spectra
        - 11fe templates as prior
        - a grid of l_t
        '''

        ln_like_spec = np.empty_like(l_t_grid, dtype=float)
        ln_like_mag = np.empty_like(l_t_grid, dtype=float)
        #ln_p = np.empty_like(l_t_grid, dtype=float)
        dic = {}
        traces = []
        for k, l_t in enumerate(l_t_grid):
            print(f'l_t = {l_t}')
            _, trace = self.MCMC_spec_phot_fixed_kernel(index_train,
                                                        index_validate,
                                                        nchains=nchains,
                                                        ndraws=ndraws,
                                                        nburns=nburns,
                                                        l_t=l_t)
            traces.append(trace)
            mag_keys = [
                key for key in trace.log_likelihood.keys() if 'Mag' in key
            ]
            ln_like_mag_temp = 0
            for key in mag_keys:
                ln_like_mag_temp = ln_like_mag_temp + trace.log_likelihood[
                    key].to_numpy()
            ln_like_spec_temp = trace.log_likelihood.lg_Delta_F0.to_numpy()
            trace.log_likelihood["sum"] = xr.DataArray(
                ln_like_spec_temp + ln_like_mag_temp,
                dims=("chain", "draw", "sum_dim"))
            print(az.loo(trace, var_name='Mag'))
            dic[f'l_t = {l_t}'] = trace

            ln_like_spec[k] = np.mean(ln_like_spec_temp.ravel())
            ln_like_mag[k] = np.mean(ln_like_mag_temp.ravel())
            #ln_p[k] = np.max(trace.sample_stats.lp.to_numpy().ravel()) log_p up to a constant (evidence)

        ln_like_spec -= ln_like_spec.max()  # remove a constant
        ln_like_mag -= ln_like_mag.max()  # remove a constant
        df_comparison = az.compare(dic, var_name='Mag')
        return traces, ln_like_spec, ln_like_mag, df_comparison

    def MCMC_spec_phot_fixed_kernel_prior(self,
                                          index_train,
                                          index_validate,
                                          l_t=None,
                                          ndraws=500,
                                          nburns=1000):
        '''
        sampling the spectrum at a given phase
        - 11fe templates as prior
        - fixed kernel parameters
        '''

        Alambda = calALambda(wv_grid0 * (1 + self.z), EBV=self.EBV, RV=3.1)

        wv_test, phase_test = np.meshgrid(wv_grid0,
                                          self.phase_spec[index_train])
        X = np.array([phase_test.ravel(), wv_test.ravel()]).T

        if l_t == None:
            with pm.Model() as Phot:
                print('Searching for MAP estimators for l_t and l_wv...')
                ln_l_t = pm.Uniform('ln_l_t', np.log(10), np.log(500))
                l_t = pm.Deterministic('l_t', np.exp(ln_l_t))
                l_w = float(self.prior.gp_res['l_wv'])
                A = 1  #float(self.prior.gp_res['A'])

                # setting up the GP model
                cov = A * pm.gp.cov.Matern32(input_dim=2, ls=[l_t, l_w])
                gp = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov)

                # GP spectra v.s. observed spectra
                lg_f0 = gp.marginal_likelihood(
                    'lg_f0',
                    X=np.atleast_2d(X),
                    y=self.lgfl_obj[index_train].ravel(),
                    noise=self.unc_obj[index_train].ravel())
                res_2D_GP = pm.find_MAP()
                print(float(res_2D_GP['l_t']),
                      float(self.prior.gp_res['l_wv']))
            l_t = float(res_2D_GP['l_t'])

        with pm.Model() as Spec_phot:

            # MAP model for 11fe
            with self.prior.gp_model:
                lg_f_11fe_mean, lg_f_11fe_cov = self.prior.gp.predict(
                    Xnew=np.atleast_2d(X), point=self.prior.gp_res)

            # setting up the GP model
            cov = pm.gp.cov.Matern32(
                input_dim=2, ls=[l_t, float(self.prior.gp_res['l_wv'])])
            gp = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov)

            # GP spectra v.s. observed spectra
            lg_Delta_f0 = gp.marginal_likelihood(
                'lg_Delta_F0',
                X=np.atleast_2d(X),
                y=self.lgfl_obj[index_train].ravel() - lg_f_11fe_mean,
                noise=self.unc_obj[index_train].ravel())

            # synthetic photometry v.s. light curve
            Xnew = np.array(
                [np.repeat(self.phase_spec[index_validate], L_wv), wv_grid0]).T
            lg_Delta_f = gp.conditional('lg_Delta_F', Xnew=np.atleast_2d(Xnew))
            with self.prior.gp_model:
                lg_f_11fe_mean_temp, lg_f_11fe_cov_temp = self.prior.gp.predict(
                    Xnew=np.atleast_2d(Xnew), point=self.prior.gp_res)
            lg_Flambda = pm.Deterministic('lg_F_lambda',
                                          lg_Delta_f + lg_f_11fe_mean_temp)
            for j in range(2):
                lc = self.obj_lc['gr'[j]]
                phase_lc = (lc['MJD'].to_numpy() - self.MJD_max) / (1 + self.z)
                mag_mean_temp, mag_var_temp = self.gp_lc[j].predict(
                    lc['mag'].to_numpy(),
                    self.phase_spec[index_validate],
                    return_var=True)
                mag_unc_temp = mag_var_temp**.5

                Mag = pm.Deterministic(
                    'Mag_{}'.format('gri'[j]),
                    spec_to_mag(spec=[
                        wv_grid0 * (1 + self.z),
                        10**(lg_Flambda - 0.4 * Alambda),
                        np.ones(L_wv)
                    ],
                                flt=np.loadtxt(self.flts[j]))[0])
            K_g = pm.Deterministic(
                'K(g)',
                self.K_corr(wv=wv_grid0,
                            Flambda=10**(lg_Flambda - 0.4 * Alambda),
                            flt=np.loadtxt(self.flts[0])))
            K_r = pm.Deterministic(
                'K(r)',
                self.K_corr(wv=wv_grid0,
                            Flambda=10**(lg_Flambda - 0.4 * Alambda),
                            flt=np.loadtxt(self.flts[1])))
            K_g_r = pm.Deterministic('K(g-r)', K_g - K_r)

        with Spec_phot:
            res = pm.find_MAP()
            trace = pm.sample(chains=2,
                              tune=nburns,
                              draws=ndraws,
                              return_inferencedata=True)
        return res, trace

    def MCMC_spec_phot_flat_prior_fixed_kernel(self,
                                               index_train,
                                               index_validate,
                                               l_t=None):
        '''
        sampling the spectrum at a given phase
        - flat prior
        - fixed kernel parameters
        '''

        Alambda = calALambda(wv_grid0 * (1 + self.z), EBV=self.EBV, RV=3.1)

        wv_test, phase_test = np.meshgrid(wv_grid0,
                                          self.phase_spec[index_train])
        X = np.array([phase_test.ravel(), wv_test.ravel()]).T

        if l_t == None:
            with pm.Model() as Phot:
                print('Searching for MAP estimators for l_t and l_wv...')
                ln_l_t = pm.Uniform('ln_l_t', np.log(10), np.log(500))
                l_t = pm.Deterministic('l_t', np.exp(ln_l_t))
                l_w = float(self.prior.gp_res['l_wv'])
                A = 1  #float(self.prior.gp_res['A'])

                # setting up the GP model
                cov = A * pm.gp.cov.Matern32(input_dim=2, ls=[l_t, l_w])
                gp = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov)

                # GP spectra v.s. observed spectra
                lg_f0 = gp.marginal_likelihood(
                    'lg_f0',
                    X=np.atleast_2d(X),
                    y=self.lgfl_obj[index_train].ravel(),
                    noise=self.unc_obj[index_train].ravel())
                res_2D_GP = pm.find_MAP()
                print(float(res_2D_GP['l_t']),
                      float(self.prior.gp_res['l_wv']))
            l_t = float(res_2D_GP['l_t'])

        with pm.Model() as Spec_phot:

            l_w = float(self.prior.gp_res['l_wv'])
            A = 1  #float(self.prior.gp_res['A'])

            # setting up the GP model
            cov = A * pm.gp.cov.Matern32(input_dim=2, ls=[l_t, l_w])
            gp = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov)

            # GP spectra v.s. observed spectra
            lg_Delta_f0 = gp.marginal_likelihood(
                'lg_F0',
                X=np.atleast_2d(X),
                y=self.lgfl_obj[index_train].ravel(),
                noise=self.unc_obj[index_train].ravel())

            # synthetic photometry v.s. light curve
            Xnew = np.array(
                [np.repeat(self.phase_spec[index_validate], L_wv), wv_grid0]).T
            lg_Flambda = gp.conditional('lg_F_lambda',
                                        Xnew=np.atleast_2d(Xnew))
            for j in range(2):
                lc = self.obj_lc['gr'[j]]
                phase_lc = (lc['MJD'].to_numpy() - self.MJD_max) / (1 + self.z)
                mag_mean_temp, mag_var_temp = self.gp_lc[j].predict(
                    lc['mag'].to_numpy(),
                    self.phase_spec[index_validate],
                    return_var=True)
                mag_unc_temp = mag_var_temp**.5

                Mag_mean = pm.Deterministic(
                    'Mag_mean_{}'.format('gri'[j]),
                    spec_to_mag(spec=[
                        wv_grid0 * (1 + self.z),
                        10**(lg_Flambda - 0.4 * Alambda),
                        np.ones(L_wv)
                    ],
                                flt=np.loadtxt(self.flts[j]))[0])
                Mag = pm.Normal('Mag_{}'.format('gri'[j]),
                                Mag_mean,
                                mag_unc_temp,
                                observed=mag_mean_temp - self.distmod)
            K_g = pm.Deterministic(
                'K(g)',
                self.K_corr(wv=wv_grid0,
                            Flambda=10**lg_Flambda,
                            flt=np.loadtxt(self.flts[0])))
            K_r = pm.Deterministic(
                'K(r)',
                self.K_corr(wv=wv_grid0,
                            Flambda=10**lg_Flambda,
                            flt=np.loadtxt(self.flts[1])))
            K_g_r = pm.Deterministic('K(g-r)', K_g - K_r)

        with Spec_phot:
            res = pm.find_MAP()
            trace = pm.sample(chains=2, draws=500, return_inferencedata=True)
        return res, trace

    def K_corr(self, wv, Flambda, flt):
        '''
        Calculate the synthetic K-correction
        '''
        mag = spec_to_mag(spec=[wv * (1 + self.z), Flambda,
                                np.ones_like(wv)],
                          flt=flt)[0]
        mag_z0 = spec_to_mag(spec=[wv, Flambda, np.ones_like(wv)], flt=flt)[0]
        return mag_z0 - mag


def optimize_gp(gps, ys):
    from scipy.optimize import minimize

    # Define the objective function (negative log-likelihood in this case).
    def nll(p):
        ll = 0
        for gp, y in zip(gps, ys):
            gp.set_parameter_vector(p)
            ll += gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    # And the gradient of the objective function.
    def grad_nll(p):
        grad = 0
        for gp, y in zip(gps, ys):
            gp.set_parameter_vector(p)
            grad += gp.grad_log_likelihood(y, quiet=True)
        return -grad

    # Print the initial ln-likelihood.
    ll = 0
    for gp, y in zip(gps, ys):
        ll += gp.log_likelihood(y, quiet=True)
    print('Initial log-likelihood:', ll)

    # Run the optimization routine.
    p0 = gps[0].get_parameter_vector()
    results = minimize(nll, p0, jac=grad_nll)

    # Update the kernel and print the final log-likelihood.
    ll = 0
    for gp, y in zip(gps, ys):
        ll += gp.log_likelihood(y, quiet=True)
    print('Maximized log-likelihood:', ll)
    return results.x
