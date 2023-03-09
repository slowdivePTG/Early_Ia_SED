'''
Inverse-variance-weighted binning
'''

import numpy as np

def data_binning(data, size=2, min_bin=1):  # size - day
    from astropy.stats import mad_std
    data_bin = []
    i = 0
    while i < len(data):
        j = i
        while j < len(data):
            if (data[j, 0] < data[i, 0] + size):
                j += 1
            else:
                break
        temp = data[i:j, :]
        if len(temp) >= min_bin:
            if len(temp) > 1:
                '''arg = np.argwhere(
                    abs(temp[:, 1] - np.median(temp[:, 1])) <= 3 *
                    mad_std(temp[:, 1])).flatten()'''
                arg = np.arange(len(temp))
                date_bin = temp[arg, 0].mean()
                weight_mag = (temp[arg, 1] / temp[arg, 2]**2) / \
                    (temp[arg, 2]**(-2)).mean()
                mag_bin = weight_mag.mean()
                '''magerr_bin = (1 / (data[arg, 2]**(-2)).mean() /
                              len(data[arg, 2]))**(0.5)'''
                magerr_bin = (temp[arg, 2]**2).sum()**.5/len(arg)
                i = j
            else:
                date_bin = data[i, 0]
                mag_bin = data[i, 1]
                magerr_bin = data[i, 2]
                i = i + 1
            data_bin.append([date_bin, mag_bin, magerr_bin])
        else:
            for t in temp.reshape(len(temp.flatten()) // 3, 3):
                data_bin.append(t)
            i = j
    return np.array(data_bin)


def plot_box_spec(wave, flux):
    flux_plot = np.repeat(flux, 2)
    wv_plot = wave.copy()
    wv_plot[:-1] += np.diff(wave) / 2
    wv_plot = np.append(
        wave[0] - (wave[1] - wave[0]) / 2,
        np.append(np.repeat(wv_plot[0:-1], 2),
                  wave[-1] + (wave[-1] - wave[-2]) / 2))

    return wv_plot, flux_plot