"""Time- and frequency-domain features per window (wrist signals only)."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .data import FS


def _slope(x: np.ndarray, fs: float) -> float:
    t = np.arange(len(x)) / fs
    return float(np.polyfit(t, x, 1)[0]) if len(x) > 1 else 0.0


def _band_power(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f < hi)
    return float(np.trapezoid(p[m], f[m])) if m.sum() > 1 else 0.0


def bvp_features(bvp: np.ndarray) -> dict:
    fs = FS["BVP"]
    x = bvp.ravel()
    b, a = sps.butter(3, [0.7, 3.5], btype="band", fs=fs)
    xf = sps.filtfilt(b, a, x)
    peaks, _ = sps.find_peaks(xf, distance=int(0.33 * fs), prominence=np.std(xf) * 0.3)
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]  # 40-180 bpm
    feats = {}
    if len(ibi) >= 5:
        hr = 60.0 / ibi
        d = np.diff(ibi)
        feats.update(hr_mean=hr.mean(), hr_std=hr.std(), ibi_sdnn=ibi.std(),
                     ibi_rmssd=np.sqrt(np.mean(d ** 2)), ibi_pnn50=np.mean(np.abs(d) > 0.05))
        # HRV spectrum: resample IBI series at 4 Hz, Welch
        t = np.cumsum(ibi)
        if t[-1] - t[0] > 20:
            tu = np.arange(t[0], t[-1], 0.25)
            iu = np.interp(tu, t, ibi)
            f, p = sps.welch(iu - iu.mean(), fs=4.0, nperseg=min(len(iu), 128))
            lf, hf = _band_power(f, p, 0.04, 0.15), _band_power(f, p, 0.15, 0.4)
            feats.update(hrv_lf=lf, hrv_hf=hf, hrv_lf_hf=lf / hf if hf > 0 else np.nan)
    f, p = sps.welch(xf, fs=fs, nperseg=min(len(xf), 512))
    feats.update(bvp_std=xf.std(), bvp_peak_freq=float(f[np.argmax(p)]))
    return feats


def eda_features(eda: np.ndarray) -> dict:
    fs = FS["EDA"]
    x = eda.ravel()
    b, a = sps.butter(2, 1.0, btype="low", fs=fs)
    x = sps.filtfilt(b, a, x)
    phasic = x - sps.medfilt(x, kernel_size=2 * int(4 * fs) + 1)  # ~8 s median = tonic
    scr, _ = sps.find_peaks(phasic, prominence=0.01)
    return dict(eda_mean=x.mean(), eda_std=x.std(), eda_min=x.min(), eda_max=x.max(),
                eda_range=x.max() - x.min(), eda_slope=_slope(x, fs),
                eda_phasic_std=phasic.std(), eda_scr_rate=len(scr) / (len(x) / fs / 60))


def temp_features(temp: np.ndarray) -> dict:
    x = temp.ravel()
    return dict(temp_mean=x.mean(), temp_std=x.std(), temp_slope=_slope(x, FS["TEMP"]))


def acc_features(acc: np.ndarray) -> dict:
    fs = FS["ACC"]
    mag = np.linalg.norm(acc, axis=1)
    f, p = sps.welch(mag - mag.mean(), fs=fs, nperseg=min(len(mag), 256))
    return dict(acc_mag_mean=mag.mean(), acc_mag_std=mag.std(),
                acc_bp_0_3=_band_power(f, p, 0.0, 3.0), acc_bp_3_8=_band_power(f, p, 3.0, 8.0),
                acc_peak_freq=float(f[np.argmax(p)]),
                **{f"acc_{ax}_std": acc[:, i].std() for i, ax in enumerate("xyz")})


def window_features(sig: dict) -> dict:
    return {**bvp_features(sig["BVP"]), **eda_features(sig["EDA"]),
            **temp_features(sig["TEMP"]), **acc_features(sig["ACC"])}
