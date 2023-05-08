import scipy.io
import glob
import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from pandas import DataFrame
import matplotlib
from progress.bar import Bar
from alive_progress import alive_bar

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings


def LoadTroikaDataset():
    """
    Retrieve the .mat filenames for the troika dataset.

    Review the README in ./datasets/troika/ to understand the organization of the .mat files.

    Returns:
        data_fls: Names of the .mat files that contain signal data
        ref_fls: Names of the .mat files that contain reference data
        <data_fls> and <ref_fls> are ordered correspondingly, so that ref_fls[5] is the
            reference data for data_fls[5], etc...
    """
    data_dir = "/Users/marcohidalgo/Desktop/training_data"
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls


def LoadTroikaDataFile(data_fl):
    """
    Loads and extracts signals from a troika data file.

    Usage:
        data_fls, ref_fls = LoadTroikaDataset()
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fls[0])

    Args:
        data_fl: (str) filepath to a troika .mat file.

    Returns:
        numpy arrays for ppg, accx, accy, accz signals.
    """
    data = sp.io.loadmat(data_fl)['sig']
    return data[2:]


def AggregateErrorMetric(pr_errors, confidence_est):
    """
    Computes an aggregate error metric based on confidence estimates.

    Computes the MAE at 90% availability.

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding
            reference heart rates.
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        the MAE at 90% availability
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estimates = pr_errors[confidence_est >= percentile90_confidence]

    # Return the mean absolute error
    print(np.mean(np.abs(best_estimates)))
    return np.mean(np.abs(best_estimates))


def psd(data, fs):
    nperseg = int(len(data) / 3)
    #psd_, freq, _, _ = plt.specgram(data, Fs=fs, NFFT=nperseg * 3,
    #                                noverlap=int(nperseg / 2))

    freq, psd_ = signal.welch(x=data, fs=fs, window="hann", nperseg=nperseg,
                                     nfft=nperseg* 8, noverlap=int(nperseg / 2))

    peaks_idx = sp.signal.find_peaks(psd_, height=0.05 * np.max(psd_), distance=1.3)[0]
    freq_use = (freq > 40 / 60) & (freq < 240 / 60)
    psd_[~freq_use] = 0.0
    psd_sorted = peaks_idx[np.argsort(psd_[peaks_idx], axis=0)[::-1]]

    #freq_use = (freq > 40 / 60) & (freq < 240 / 60)
    #psd_[~freq_use] = 0.0
    #psd_sorted = (-peaks_idx).argsort(axis=0)[:4]

    return psd_, freq, psd_sorted


# Frequency features
def frequency_features(epoch_data, **args):
    # FFT the ppg signal
    fs = args['fs']
    # nperseg_ppg = int(len(epoch_data)/3)
    # nperseg_acc = int(len(epoch_data)/4)

    # _, psd_residual = signal.welch(x=epoch_data['residual'], fs=fs, window="hann", nperseg=nperseg_ppg,nfft=nperseg_ppg * 8, noverlap=int(nperseg_ppg / 2))
    # freq_ppg, psd_ppg = signal.welch(x=epoch_data['ppg'], fs=fs, window="hann", nperseg=nperseg_ppg, nfft=nperseg_ppg*8, noverlap=int(nperseg_ppg / 2))
    # freq_acc, psd_acc = signal.welch(x=epoch_data['accmag'], fs=fs, window="hann", nperseg=nperseg_acc, nfft=nperseg_acc * 8, noverlap=int(nperseg_acc / 5))
    # freq_acc, psd_acc = signal.welch(x=epoch_data[['accmag', 'accx', 'accy', 'accz']], fs=fs, window="hann", nperseg=nperseg_acc,
    #                                 nfft=nperseg_acc * 8, noverlap=int(nperseg_acc / 3),axis=0)

    # plotting = False
    spec_ppg, freq_ppg, sorted_ppg = psd(epoch_data['ppg'], fs=fs)
    spec_accx, _, sorted_accx = psd(epoch_data['accx'], fs=fs)
    spec_accy, _, sorted_accy = psd(epoch_data['accy'], fs=fs)
    spec_accz, _, sorted_accz = psd(epoch_data['accz'], fs=fs)

    freqs = freq_ppg[sorted_ppg]
    cond1 = freq_ppg[sorted_accx[0]]
    cond2 = freq_ppg[sorted_accy[0]]
    cond3 = freq_ppg[sorted_accz[0]]

    for freq_ in freqs:
        if freq_ == cond1 or freq_ == cond1 or freq_ == cond3:
            continue
        else:
            use_peak = freq_

    #   # Find peaks
    #   peaks_ppg_idx = sp.signal.find_peaks(psd_ppg, height=0.05 * np.max(psd_ppg), distance=1.3)[0]# height=0.05 * np.max(psd_ppg)
    #   peaks_acc_idx = sp.signal.find_peaks(psd_residual, height=0.05 * np.max(psd_residual), distance=1.3)[0]

    #   # Estimates sorted by power
    #   estimates_freq = freq_ppg[peaks_ppg_idx][np.argsort(psd_ppg[peaks_ppg_idx], axis=0)[::-1]]
    #   estimate_psd = psd_ppg[peaks_ppg_idx][np.argsort(psd_ppg[peaks_ppg_idx], axis=0)[::-1]]

    #   artifact_freq = freq_ppg[peaks_acc_idx][np.argsort(psd_residual[peaks_acc_idx], axis=0)[::-1]]
    #   artifact_psd = psd_residual[peaks_acc_idx][np.argsort(psd_residual[peaks_acc_idx], axis=0)[::-1]]

    #   artifact_freq = artifact_freq[:2]
    #   artifact_psd = artifact_psd[:2]
    #   # Window limit to remove outliers
    #   low_freqs_ppg = (estimates_freq >= (40 / 60)) & (estimates_freq <= (200/ 60))
    #   estimate_window = estimate_psd[low_freqs_ppg]
    #   freqs_estimate_window = estimates_freq[low_freqs_ppg]

    #   use_peak = freqs_estimate_window[0]
    # if len(freqs_estimate_window) == 1:
    #    use_peak = freqs_estimate_window[0]
    # elif len(freqs_estimate_window) == 2:
    #    use_peak = np.delete(freqs_estimate_window, np.where(min(abs(freqs_estimate_window - artifact_freq[0])))[0])[0]
    #   if abs(freqs_estimate_window[0] - artifact_freq[0]) < 1:
    #       use_peak = freqs_estimate_window[-1]

    # plt.plot(freq_ppg, psd_ppg)
    # plt.plot(freq_acc, psd_acc)
    # plt.plot(freqs_estimate_window, estimate_window, '*')
    # plt.plot(artifact_freq, artifact_psd, 'o')
    # plt.text(3, np.max(psd_ppg), str(use_peak*60))
    # plt.axvline(x=use_peak)
    # plt.xlim([0, 10])
    # plt.show()
    selected = use_peak * 60
    print(selected)
    # Confidence
    secs = 10  # Only 10 seconds
    win = (secs / 60.0)
    win_freqs = (freq_ppg >= selected / 60 - win) & (freq_ppg <= selected / 60 + win)
    confidence = np.sum(spec_ppg[win_freqs]) / np.sum(spec_ppg)

    results = {'BPM': [selected], 'Confidence': [confidence]}
    features_df = pd.DataFrame.from_dict(results)
    return features_df


class Data():
    def __init__(self, data: DataFrame, fs: float):
        self.data = data
        self.fs = fs

    def data(self):
        return self.data

    def describe(self):
        self.summary = self.data.describe().loc[['mean', 'std'], :]

    def normalize(self):
        mu = self.summary.loc['mean', :]
        sigma = self.summary.loc['std', :]
        self.data_norm = ((self.data_filtered - mu) / sigma)
        mu_norm = self.data_norm.describe().loc['mean', :]
        self.data_norm = self.data_norm - mu_norm

    def rls(self, n=1, eps=0.001, ff=0.989):
        w = np.random.normal(0, 0.5, n)
        N = len(self.data_norm)
        R = 1 / eps * np.identity(n)
        y = np.zeros(N)
        e = np.zeros(N)

        d = self.data_norm['ppg'].values
        x = self.data_norm[['accx', 'accy', 'accz']].values
        # adaptation loop
        for k in range(N):
            y[k] = np.dot(w, x[k])
            e[k] = d[k] - y[k]
            R1 = np.dot(np.dot(np.dot(R, x[k]), x[k].T), R)
            R2 = ff + np.dot(np.dot(x[k], R), x[k].T)
            R = 1 / ff * (R - R1 / R2)
            dw = (np.dot(R, x[k].T) * e[k])
            w += dw
        self.data_norm['ppg'] = e
        self.data_norm['residual'] = y

    def filter_(self, order):
        minutes = 60
        low_cut, high_cut = 40 / minutes, 240 / minutes
        b, a = sp.signal.butter(order, (low_cut, high_cut), btype='bandpass', fs=self.fs)
        self.data_filtered = DataFrame(sp.signal.filtfilt(b, a, self.data, axis=0), columns=self.data.columns)
        return self.data_filtered

    def compute_hr(self, window_size_sec: int, overlap_sec: int):
        self.win_size = np.round(window_size_sec * self.fs).astype(int)
        self.overlap = np.round(overlap_sec * self.fs).astype(int)
        self.history = np.array([])

        assert len(self.data_norm) >= self.win_size, "Window size should be less of equal than data"
        shift = self.win_size - self.overlap
        n_row, n_cols = self.data_norm.shape

        features = []
        features.append(pd.concat([self.hr_estimation(self.data_norm.iloc[i:i + self.win_size]) for i in
                                   range(0, n_row - self.win_size, shift)]))
        return features

    def hr_estimation(self, data):
        spec_ppg, freq_ppg, sorted_ppg = psd(data['ppg'], fs=self.fs)
        spec_accx, _, sorted_accx = psd(data['accx'], fs=self.fs)
        spec_accy, _, sorted_accy = psd(data['accy'], fs=self.fs)
        spec_accz, _, sorted_accz = psd(data['accz'], fs=self.fs)

        freqs = freq_ppg[sorted_ppg]
        cond1 = freq_ppg[sorted_accx[0]]
        cond2 = freq_ppg[sorted_accy[0]]
        cond3 = freq_ppg[sorted_accz[0]]

        if len(self.history) == 0:
            previous = freqs[0]
            idx_from_prev_hr = np.arange(0, len(freqs), 1)
        else:
            previous = self.history[-1]
            idx_from_prev_hr = np.argsort(np.abs(freqs - previous), axis=0).squeeze()

        if len(freqs) == 1:
            current_hr = freqs
            n = abs(int(np.abs(current_hr - previous)))
            if n > 1:
                current_hr = np.median(self.history)

        else:
            #previous = freqs[0]
            #current_hr = previous#freqs[0]
            for freq_ in freqs[idx_from_prev_hr]:
                current_hr = freq_
                #current_hr = freq_
                if current_hr == cond1 or current_hr == cond2 or current_hr == cond3:
                    pass
                else:
                    n = np.abs(int(np.abs(current_hr - previous)))
                    if n > 1:
                        current_hr = np.mean(self.history)
                    else:
                        current_hr = freq_
                        break

        self.history = np.append(self.history, current_hr)
        selected = current_hr.squeeze() * 60
        # Confidence
        secs = 20  # Only 10 seconds
        win = (secs / 60.0)
        win_freqs = (freq_ppg >= selected / 60 - win) & (freq_ppg <= selected / 60 + win)
        confidence = np.sum(spec_ppg[win_freqs.T]) / np.sum(spec_ppg)

        results = {'BPM': [selected], 'Confidence': [confidence]}
        features_df = pd.DataFrame.from_dict(results)
        return features_df


def RunPulseRateAlgorithm(data_fl, ref_fl):
    # Load data using LoadTroikaDataFile
    ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)

    gold_standard = sp.io.loadmat(ref_fl)['BPM0']
    accmag = np.sqrt(accx ** 2 + accy ** 2 + accz ** 2)
    ppg_accmagdf = pd.DataFrame(np.vstack((ppg, accmag, accx, accy, accz)).T,
                                columns={'ppg', 'accmag', 'accx', 'accy', 'accz'})

    window = 8
    ovelap = 6
    fs = 125
    ppgObj = Data(ppg_accmagdf, fs=fs)
    ppgObj.filter_(order=2)
    ppgObj.describe()
    ppgObj.normalize()
    ppgObj.rls(n=3)
    bpm_estimate = ppgObj.compute_hr(window_size_sec=window, overlap_sec=ovelap)
    algorithm_estimate = bpm_estimate[0]['BPM'].values
    confidence = bpm_estimate[0]['Confidence'].values.tolist()

    # Correct for differences
    if len(gold_standard) != len(algorithm_estimate):
        if len(gold_standard) < len(algorithm_estimate):
            algorithm_estimate = algorithm_estimate[:len(gold_standard)]
            confidence = confidence[:len(gold_standard)]
        elif len(gold_standard) > len(algorithm_estimate):
            gold_standard = gold_standard[:len(algorithm_estimate)]

    errors = np.abs(gold_standard - algorithm_estimate)[0]
    return errors.tolist(), confidence


def Evaluate():
    """
        Top-level function evaluation function.

        Runs the pulse rate algorithm on the Troika dataset and returns an aggregate error metric.

        Returns:
            Pulse rate error on the Troika dataset. See AggregateErrorMetric.
        """
    # Retrieve dataset files
    data_fls, ref_fls = LoadTroikaDataset()
    errs, confs = [], []
    with alive_bar(200, bar='smooth', spinner='notes2') as bar:
        for data_fl, ref_fl in zip(data_fls, ref_fls):
            # Run the pulse rate algorithm on each trial in the dataset
            errors, confidence = RunPulseRateAlgorithm(data_fl, ref_fl)
            errs.append(errors)
            confs.append(confidence)
            bar()
    errs = np.hstack(errs)
    confs = np.hstack(confs)
    return AggregateErrorMetric(errs, confs)


if __name__ == "__main__":
    Evaluate()
