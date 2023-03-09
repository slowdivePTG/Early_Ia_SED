import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import arviz as az

import tinygp
from tinygp import kernels, GaussianProcess, transforms
from tinygp.kernels.distance import L2Distance

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
numpyro.set_host_device_count(2)

import jax
import jax.numpy as jnp
import jaxopt
from jax import lax

jax.config.update("jax_enable_x64", True)

from scipy.interpolate import interp1d

import sys

sys.path.append('./tools/')
from dust_extinction import calALambda
from data_binning import data_binning
from spec_to_flux import spec_to_mag

## 2D grid
## phase: -20 to 10 days, step = 1 day
## wavelength: 3750 to 7500 Ang, step = 25 Ang

phase_grid0 = np.arange(start=-20, stop=11)
wv_grid0 = np.arange(start=3750, stop=7525, step=25)
L_phase = len(phase_grid0)
L_wv = len(wv_grid0)
phase_grid, wv_grid = np.meshgrid(phase_grid0, wv_grid0)


class SN_phot_spec:

    def __init__(self,
                 obj_lc,
                 obj_spec_meta,
                 obj_spec_files,
                 flts,
                 prior,
                 SN_name=None,
                 savefig=False,
                 plot_syn_phot=True):
        '''
        Parameters
        ----------
        obj_lc : dict
            multiband light curves
            each element is a dataframe
            columns - MJD, mag, mag_unc
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
        self.SN_name = SN_name
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

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
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
            gp_temp = build_gp(params={
                'log_scale': np.log(20),
                'mean': 18.5
            },
                               X=phase_lc.values.reshape(-1, 1),
                               yerr=lc['mag_err'].values)
            gp_lc.append(gp_temp)
            t_grid = np.linspace(-20, 10, 100)
            mag_grid, mag_var_grid = gp_temp.predict(lc['mag'].values,
                                                     np.atleast_2d(t_grid).T,
                                                     return_var=True)
            ax[0].fill_between(t_grid,
                               mag_grid - self.distmod - mag_var_grid**.5,
                               mag_grid - self.distmod + mag_var_grid**.5,
                               color=cols[j],
                               alpha=.3)
            # calculate effective wavelength given the throughput function
            T = np.loadtxt(self.flts[j])
            wv_eff = np.append(wv_eff,
                               (T[:, 0] * T[:, 1]).sum() / T[:, 1].sum())

        self.gp_lc = gp_lc
        self.wv_eff = wv_eff
        ax[0].invert_yaxis()
        ax[0].legend(prop={'size': 20})

        phase_spec = []
        spec_fl, spec_fl_unc = [], []
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
                mag_fit, mag_fit_var = gp_lc[j].predict(lc['mag'].values,
                                                        np.atleast_2d(phase).T,
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
                    np.atleast_2d(phase).T,
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
            Fl_interp = fl_interp * (dis / 10)**2 * 10**(0.4 * calALambda(
                wv_grid0 * (1 + self.z), EBV=self.EBV, RV=3.1)) * (1 + self.z)
            unc_interp = np.min(Fl_interp) * np.ones_like(
                Fl_interp) * 0.1  # assuming SNR >= 10 after binning

            lgfl_11fe, lgfl_11fe_unc = prior.get_spec(phase)
            spec_fl.append(Fl_interp)
            spec_fl_unc.append(unc_interp)

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

        # mags from light curves
        self.lc_mag = {}
        self.lc_mag_unc = {}
        self.lc_f_lam = {}  # f_lambda : erg cm^-2 s^-1 AA^-1
        self.lc_f_lam_unc = {}
        self.phase_lc = {}
        for j in range(2):
            lc = self.obj_lc['gr'[j]]
            phase_lc = (lc['MJD'].values - self.MJD_max) / (1 + self.z)
            mag, mag_unc = lc['mag'].values, lc['mag_err'].values
            mag = mag - self.distmod - calALambda(
                self.wv_eff[j], EBV=self.EBV, RV=3.1)
            f_nu = 10**(-0.4 * mag) * 3631 * 1e-23  # cgs
            dlambda_dnu = self.wv_eff[j] / (2.99792458e10 /
                                            (self.wv_eff[j] * 1e-8))  # Ang/Hz
            f_lam = f_nu / dlambda_dnu
            f_lam_unc = f_lam / 2 * (10**(0.4 * mag_unc) -
                                     10**(-0.4 * mag_unc))

            phase_index = (phase_lc <= np.array(phase_spec).max()) & (
                mag_unc < 0.2)  # exclude late-type photometry
            self.phase_lc['gr'[j]] = phase_lc[phase_index]
            self.lc_mag['gr'[j]] = mag[phase_index]
            self.lc_mag_unc['gr'[j]] = mag_unc[phase_index]
            self.lc_f_lam['gr'[j]] = f_lam[phase_index]
            self.lc_f_lam_unc['gr'[j]] = f_lam_unc[phase_index]

        ax[0].set_yticks([-16, -16.5, -17, -17.5, -18, -18.5, -19, -19.5])
        ax[0].set_xticks([-20, -10, 0, 10])
        ax[1].set_yticks([])
        # ax[0].set_xlabel(r'$t-t_{\mathrm{max},B}\ [\mathrm{day}]$')
        # ax[0].set_ylabel(r'$M$')
        # ax[1].set_ylabel(r'$\log F_\lambda+\mathrm{offset}$')
        # ax[1].set_xlabel(r'$\lambda$')

        ax[0].set_xlim(-22, 11)
        ax[0].set_ylim(-15.8, -19.6)
        ax[1].set_xlim(3300, 8400)
        if savefig:
            plt.savefig('Figure/{}_basic.pdf'.format(self.SN_name),
                        bbox_inches='tight')

        self.phase_spec = np.array(phase_spec)
        self.spec_fl = np.array(spec_fl)
        self.spec_fl_unc = np.array(spec_fl_unc)

        # get weights
        # w(lambda, t) = T(lambda) * F_mean(lambda, t) / sum(T(lambda) * F_mean(lambda, t))
        self.w = {
            'g': np.ones((len(self.phase_lc['g']), L_wv)),
            'r': np.ones((len(self.phase_lc['r']), L_wv))
        }
        for j in range(2):
            flt = 'gr'[j]
            T0 = np.loadtxt(self.flts[j])
            T = interp1d(x=T0[:, 0] / (1 + self.z),
                         y=T0[:, 1],
                         fill_value=0,
                         bounds_error=False)(wv_grid0)
            for k, phase in enumerate(self.phase_lc[flt]):
                F_prior_mean, _ = self.prior.get_spec(phase, get_log=False)
                wv_T_F_prior_mean = wv_grid0 * T * F_prior_mean
                self.w[flt][k] = wv_T_F_prior_mean / wv_T_F_prior_mean.sum()

    def MCMC_spec_lc(self,
                     index_train,
                     index_validate,
                     bin_size=2,
                     train=False,
                     train_MCMC=False,
                     savefig=False):
        '''
        sampling the observed light curves & spectra
        - model the joint distribution of flux in g, r, and spectra (wrt to a mean model)
            - cov matrix between spectra - Matern32
            - cov matrix between g/r fluxes and spectra - Matern32 * w -> w = the weighted throughput vector
            - cov matrix between fluxes - w^T * Matern32 * w
        - 11fe templates as the mean model
        
        Parameters
        -------------------
        index_train: array-like
            Phase indices of the spectra used in the GP fitting
        '''

        wv_spec, phase_spec = np.meshgrid(wv_grid0,
                                          self.phase_spec[index_train])

        # coordinates - [phase, wv, phase_idx]
        # for spec - phase_idx = 1
        # for phot - phase = self.phase_lc[phase_idx]
        X_spec = np.array([
            phase_spec.ravel(),
            wv_spec.ravel(), -np.ones_like(phase_spec.ravel())
        ]).T
        # wv == -2 --> g flux
        X_g = np.array([
            self.phase_lc['g'],
            np.repeat(-2, len(self.phase_lc['g'])),
            np.arange(len(self.phase_lc['g']))
        ]).T
        # wv == -1 --> r flux
        X_r = np.array([
            self.phase_lc['r'],
            np.repeat(-1, len(self.phase_lc['r'])),
            np.arange(len(self.phase_lc['r']))
        ]).T
        X = np.concatenate([X_spec, X_g, X_r])

        # spectral flux density
        F = self.spec_fl[index_train]
        F_unc = self.spec_fl_unc[index_train]
        F_prior_mean = F.copy()
        for k, phase in enumerate(self.phase_spec[index_train]):
            F_prior_mean[k], _ = self.prior.get_spec(phase,
                                                     get_cov=False,
                                                     get_log=False)
        xi_spec = (F / F_prior_mean).ravel()
        xi_unc_spec = (F_unc / F_prior_mean).ravel()

        # photometry
        f_g = self.lc_f_lam['g']
        f_g_unc = self.lc_f_lam_unc['g']
        f_prior_mean = f_g.copy()
        for k, phase in enumerate(self.phase_lc['g']):
            f_prior_mean[k], _ = self.prior.get_syn_flux(phase,
                                                         flt=np.loadtxt(
                                                             self.flts[0]),
                                                         z=self.z)
        xi_g = (f_g / f_prior_mean).ravel()
        xi_unc_g = (f_g_unc / f_prior_mean).ravel()

        f_r = self.lc_f_lam['r']
        f_r_unc = self.lc_f_lam_unc['r']
        f_prior_mean = f_r.copy()
        for k, phase in enumerate(self.phase_lc['r']):
            f_prior_mean[k], _ = self.prior.get_syn_flux(phase,
                                                         flt=np.loadtxt(
                                                             self.flts[1]),
                                                         z=self.z)
        xi_r = (f_r / f_prior_mean).ravel()
        xi_unc_r = (f_r_unc / f_prior_mean).ravel()

        # plt.figure(figsize=(8, 6))
        # plt.errorbar(self.phase_lc['g'], xi_g, yerr=xi_unc_g, fmt='o')
        # plt.errorbar(self.phase_lc['r'], xi_r, yerr=xi_unc_r, fmt='o')
        # plt.xlabel('Phase')
        # plt.ylabel('f/f_prior')
        # plt.xlim(self.phase_spec[index_validate] - 5,
        #          self.phase_spec[index_validate] + 5)
        # plt.axvline(self.phase_spec[index_validate], color='k')
        # plt.show()

        xi = np.concatenate([xi_spec, xi_g, xi_r])
        xi_unc = np.concatenate([xi_unc_spec, xi_unc_g, xi_unc_r])

        # GP - xi(phase, wv)
        from astropy.stats import mad_std
        kernel_amp = max(max(mad_std(xi_g), mad_std(xi_r)), mad_std(xi_spec))
        kernel_mean = xi_g.ravel().mean()
        params = {
            "log_amp": np.log(kernel_amp),
            "log_scale": np.log([10, 500]),
            "mean": np.float64(kernel_mean),
        }

        def build_integrated_kernel(params):
            kernel0 = jnp.exp(params['log_amp'] * 2) * transforms.Linear(
                jnp.exp(-params['log_scale']),
                kernel=kernels.Matern32(distance=L2Distance()))
            kernel = IntegratedKernel(kernel=kernel0,
                                      weights=self.w,
                                      bin_size=bin_size)
            gp = GaussianProcess(kernel=kernel,
                                 X=X,
                                 diag=xi_unc**2,
                                 mean=params['mean'])
            return gp

        if not train:
            gp = build_integrated_kernel(params)
        elif not train_MCMC:

            @jax.jit
            def loss(params):
                return -build_integrated_kernel(params).log_probability(xi)

            print('Initial log_p = {:.1f}'.format(-loss(params)))
            solver = jaxopt.ScipyMinimize(fun=loss)
            soln = solver.run(params)
            print(np.exp(soln.params["log_amp"]),
                  np.exp(soln.params["log_scale"]), soln.params["mean"])
            print('Final log_p = {:.1f}'.format(-loss(soln.params)))
            gp = build_integrated_kernel(soln.params)

        else:

            def numpyro_model():
                log_amp = numpyro.sample("log_amp",
                                         dist.Normal(params['log_amp'], 0.1))
                log_scale = numpyro.sample(
                    "log_scale", dist.Normal(params['log_scale'], 0.1))
                mean = numpyro.sample("mean", dist.Normal(params["mean"], 1))
                kernel0 = jnp.exp(log_amp * 2) * transforms.Linear(
                    jnp.exp(-log_scale),
                    kernel=kernels.Matern32(distance=L2Distance()))
                kernel = IntegratedKernel(kernel=kernel0,
                                          weights=self.w,
                                          bin_size=bin_size)
                gp = GaussianProcess(kernel=kernel,
                                     X=X,
                                     diag=xi_unc**2,
                                     mean=mean)
                numpyro.sample("gp", gp.numpyro_dist(), obs=xi)

            sampler = MCMC(
                NUTS(numpyro_model),
                num_warmup=500,
                num_samples=500,
                num_chains=2,
                progress_bar=True,
            )
            rng_key = jax.random.PRNGKey(114514)
            sampler.run(rng_key)

            data = az.from_numpyro(sampler)
            summary = az.summary(
                data,
                var_names=[v for v in data.posterior.data_vars if v != "pred"])
            print(summary)

            import corner
            fig = corner.corner(
                data,
                var_names=[v for v in data.posterior.data_vars if v != "pred"])
            for ax in fig.axes:
                ax.xaxis.set_tick_params(labelsize=15)
                ax.yaxis.set_tick_params(labelsize=15)
            if savefig:
                fig.savefig('Figure/{}_corner.pdf'.format(self.SN_name),
                            bbox_inches='tight')

            params_numpyro = {
                "log_amp":
                jnp.float64(summary['mean']['log_amp']),
                "log_scale":
                jnp.array([
                    summary['mean']['log_scale[0]'],
                    summary['mean']['log_scale[1]']
                ]),
                "mean":
                summary['mean']['mean']
            }
            gp = build_integrated_kernel(params_numpyro)

        wv_test, phase_test = np.meshgrid(wv_grid0,
                                          self.phase_spec[index_validate])

        # coordinates - [phase, wv, phase_idx]
        # for spec - phase_idx = 1
        # for phot - phase = self.phase_lc[phase_idx]
        X_test = np.array([
            phase_test.ravel(),
            wv_test.ravel(), -np.ones_like(phase_test.ravel())
        ]).T
        xi_pred, xi_cov_pred = gp.predict(xi, X_test, return_cov=True)
        xi_unc_pred = np.diag(xi_cov_pred)**.5

        F_prior_mean_pred, _ = self.prior.get_spec(phase_test.ravel()[0],
                                                   get_cov=False,
                                                   get_log=False)
        F_pred = F_prior_mean_pred * xi_pred
        F_unc_pred = F_prior_mean_pred * xi_unc_pred

        fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        ax[0].plot(wv_grid0, xi_pred)
        ax[0].fill_between(wv_grid0,
                           xi_pred - xi_unc_pred,
                           xi_pred + xi_unc_pred,
                           alpha=.2)
        ax[0].plot(wv_grid0,
                   self.spec_fl[index_validate].ravel() / F_prior_mean_pred,
                   color='k')
        ax[1].plot(wv_grid0, F_pred)
        ax[1].fill_between(wv_grid0,
                           F_pred - F_unc_pred,
                           F_pred + F_unc_pred,
                           alpha=.2)
        ax[1].plot(wv_grid0, self.spec_fl[index_validate].ravel(), color='k')
        ax[1].plot(wv_grid0, F_prior_mean_pred, color='orange')
        ax[1].set_xlabel('Wavelength')
        ax[0].set_ylabel('F/F_prior')
        ax[1].set_ylabel('F')
        ax[1].set_yscale('log')
        ax[0].set_title('Phase' +
                        '= {:.2f}'.format(self.phase_spec[index_validate][0]))
        # ax[0].set_title(r'$\mathrm{Phase}' +
        #                 '= {:.2f}$'.format(self.phase_spec[index_validate][0]))
        ax[1].set_ylim(
            min(self.spec_fl[index_validate].ravel().min(),
                F_prior_mean_pred.min()) / 1.5,
            max(self.spec_fl[index_validate].ravel().max(),
                F_prior_mean_pred.max()) * 1.5)
        plt.tight_layout()
        if savefig:
            plt.savefig('Figure/{}_fit.pdf'.format(self.SN_name),
                        bbox_inches='tight')
        return F_pred, F_unc_pred

    def K_corr(self, wv, Flambda, flt):
        '''
        Calculate the synthetic K-correction
        '''
        mag = spec_to_mag(spec=[wv * (1 + self.z), Flambda,
                                np.ones_like(wv)],
                          flt=flt)[0]
        mag_z0 = spec_to_mag(spec=[wv, Flambda, np.ones_like(wv)], flt=flt)[0]
        return mag_z0 - mag


class IntegratedKernel(tinygp.kernels.Kernel):

    def __init__(self, kernel, weights, bin_size=1):
        '''
        Parameters
        ----------
        kernel : tinygp.kernels.Kernel
            the original kernel
        weights : array-like
            w[flt](t_j, wv_k) = T[flt](wv_k) * F_template(t_j, wv_k) / sum_k(T[flt](wv_k) * F_template(t_j, wv_k))
                flt : the filter, 'g' or 'r'
                t_j : the epoch of a photometric measurement
                wv_k : wavelength
                T[flt] : the throughput function of the filter
                F_template : flux density of the template/mean model
        bin_size : int
            bin the weights & wv grid to speed up the convolution
            wv grid size * bin_size should be less than l_wv
        '''
        self.kernel = kernel
        self.w = weights
        self.bin_size = bin_size
        self.L_wv = L_wv // bin_size
        self.wv_grid = jnp.mean(wv_grid0[:self.L_wv * bin_size].reshape(
            -1, bin_size),
                                axis=1)

    def evaluate(self, X1, X2):
        '''
        Parameters
        ----------
        X1, X2 : array-like
        t, wv, idx_t = X
        t : epoch
        wv : wavelength
            wv == -2 for g band photometry
            wv == -1 for r band photometry
        idx_t : index of the epoch of a photometric point in a light curve
            weights = self.w['g'/'r'][idx_t]
        '''
        t1, wv1, idx_t1 = X1
        t2, wv2, idx_t2 = X2
        idx_t1 = jnp.array(idx_t1, dtype=int)
        idx_t2 = jnp.array(idx_t2, dtype=int)

        X1_grid = jnp.array([jnp.repeat(t1, self.L_wv), self.wv_grid]).T
        X2_grid = jnp.array([jnp.repeat(t2, self.L_wv), self.wv_grid]).T
        w1 = jnp.sum(jnp.where(
            wv1 == -2,
            jnp.asarray(self.w['g'])[idx_t1],
            jnp.asarray(self.w['r'])[idx_t1])[:self.L_wv *
                                              self.bin_size].reshape(
                                                  -1, self.bin_size),
                     axis=1)
        w2 = jnp.sum(jnp.where(
            wv2 == -2,
            jnp.asarray(self.w['g'])[idx_t2],
            jnp.asarray(self.w['r'])[idx_t2])[:self.L_wv *
                                              self.bin_size].reshape(
                                                  -1, self.bin_size),
                     axis=1)

        # spec v.s. spec
        K_FF = self.kernel.evaluate(jnp.array([[t1, wv1]]),
                                    jnp.array([[t2, wv2]]))

        # phot v.s. phot
        K_ff = lax.dot(lax.dot(w1, self.kernel(X1_grid, X2_grid)), w2.T)
        K_fF = lax.dot(w1,
                       self.kernel(X1_grid, jnp.array([[t2, wv2]])).ravel())
        K_Ff = lax.dot(
            self.kernel(jnp.array([[t1, wv1]]), X2_grid).ravel(), w2.T)
        return jnp.where(wv1 > 0, jnp.where(wv2 > 0, K_FF, K_Ff),
                         jnp.where(wv2 > 0, K_fF, K_ff))


def train_gp(nparams, build_gp_func, X, y, yerr):

    @jax.jit
    def loss(params):
        return -build_gp_func(params, X, yerr).log_probability(y)

    params = {
        "log_scale": np.log([.5, 500])[:nparams],
        "mean": np.float64(0),
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
