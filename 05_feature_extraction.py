# %%
# ............................................................................
# IMPORTING LIBRARIES
# ............................................................................
import pandas as pd
import numpy as np
import wfdb
from scipy.fft import rfft, rfftfreq
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# %%
# -----------------------------------------------------------------------------
# 1. LOADING THE CLINICAL DATASET
# -----------------------------------------------------------------------------
# Load record 116 directly from PhysioNet (MIT-BIH Arrhythmia Database).
# ann holds the reference annotations (expert-labeled beat locations).
dataset = wfdb.rdrecord('116', pn_dir='mitdb')
ann = wfdb.rdann('116', 'atr', pn_dir='mitdb')
ecg_reading = dataset.p_signal[:, 0]   # lead 1 only
fs = dataset.fs
time = np.arange(len(ecg_reading))/fs

# %%
plt.figure(figsize=(12, 4))
plt.plot(time, ecg_reading, alpha=0.6)
plt.grid(True, linestyle='--')
plt.xlabel("time")
plt.ylabel("millivolt")
plt.title("Raw ECG Signal")
plt.show()

# %%
# -----------------------------------------------------------------------------
# 2. REFERENCE ANNOTATIONS OVERLAY
# -----------------------------------------------------------------------------
# Overlay the dataset's reference annotations on the raw signal.
print(ann.sample)
print(ann.symbol)
plt.figure(figsize=(12, 4))
plt.plot(time, ecg_reading, alpha=0.6)
ann_time = ann.sample/fs
plt.scatter(
    ann_time,
    ecg_reading[ann.sample],
    color='r'
)
plt.grid(True, linestyle='--')
plt.xlabel("time")
plt.ylabel("millivolt")
plt.title("Raw ECG Signal")
plt.show()

# %%
# -----------------------------------------------------------------------------
# 3. FREQUENCY ANALYSIS (FFT)
# -----------------------------------------------------------------------------
# FFT to check the signal's frequency content before choosing filter cutoffs.
mean = ecg_reading-np.mean(ecg_reading)
set_fft = np.abs(rfft(mean))
set_fftfreq = rfftfreq(len(ecg_reading), d=1/fs)

# %%
plt.figure(figsize=(10, 4))
plt.plot(set_fftfreq, set_fft, color='red')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Frequency(Hz)", fontsize=11)
plt.ylabel("Magnitude", fontsize=11)
plt.title("(ECG Spectrum Analysis)", fontsize=12)
plt.show()

# %%
# -----------------------------------------------------------------------------
# 4. BANDPASS FILTERING
# -----------------------------------------------------------------------------
# Bandpass filter (0.5-30 Hz): removes baseline wander and muscle noise.
# sosfiltfilt gives zero-phase filtering, so R-peak timing isn't shifted.
lowcut, highcut = 0.5, 30.0
order = 4
sos_butter = signal.butter(
    order,
    [lowcut, highcut],
    btype='band',
    fs=fs,
    output='sos')
filtered_butter = signal.sosfiltfilt(sos_butter, ecg_reading)

# %%
# -----------------------------------------------------------------------------
# 5. R-PEAK DETECTION
# -----------------------------------------------------------------------------
# Median/percentile-based thresholds are more robust to outliers than
# mean-based ones; prominence helps reject P/T waves.
baseline = np.median(filtered_butter)
upperline = np.percentile(filtered_butter, 80)
prominence = 0.2*(upperline-baseline)
peaks, _ = find_peaks(filtered_butter,
                      height=upperline,
                      distance=int(0.33*fs),
                      prominence=prominence)
print(f"R-peak: {len(peaks)}")
print(f"Total signal length: {len(ecg_reading)/fs:.1f} s")

# %%
# -----------------------------------------------------------------------------
# 6. RR INTERVALS
# -----------------------------------------------------------------------------
# Time between consecutive R-peaks, the base signal for arrhythmia analysis.
RR_interval = np.diff(peaks)/fs
print(f"RR_interval: {RR_interval}")
for i in range(1, len(peaks) - 1):
    RR_pre = (peaks[i] - peaks[i-1]) / fs
    RR_post = (peaks[i+1] - peaks[i]) / fs
    print(f" beat:{i}, RR_pre:{RR_pre}, RR_post:{RR_post}")

# %%
# -----------------------------------------------------------------------------
# 7. EARLY-BEAT CANDIDATES
# -----------------------------------------------------------------------------
# Beats arriving earlier than 75% of the median RR are flagged as
# premature-beat candidates (timing criterion for PVC).
RR_median = np.median(RR_interval)
early_candidates = np.where(RR_interval < 0.75 * RR_median)[0] + 1

# %%
# -----------------------------------------------------------------------------
# 8. BEAT SEGMENTATION - OPTIMIZED
# -----------------------------------------------------------------------------
# Segment an 80 ms window around each R-peak. Preallocated array is used
# instead of repeated appends for speed.
window_sample = int(0.08 * fs)
peaks_arr = np.asarray(peaks)
valid_mask = (peaks_arr - window_sample >= 0) & (peaks_arr +
                                                 window_sample < len(filtered_butter))
valid_peak_positions = peaks_arr[valid_mask]
valid_indices = np.where(valid_mask)[0]

# Maps original peak index -> position in the beats array (some peaks near
# the signal edges are dropped for lacking a full window).
idx_to_beatpos = {orig_idx: beat_pos for beat_pos,
                  orig_idx in enumerate(valid_indices)}
n_beats = len(valid_peak_positions)
beat_len = 2 * window_sample
beats = np.empty((n_beats, beat_len))
for b, p in enumerate(valid_peak_positions):
    beats[b] = filtered_butter[p - window_sample: p + window_sample]

# %%
# -----------------------------------------------------------------------------
# 9. NORMAL BEAT TEMPLATE
# -----------------------------------------------------------------------------
# Median beat shape built from all non-early beats, used as the "normal"
# QRS template for morphology comparison.
early_candidates_set = set(early_candidates.tolist())
normal_mask = np.ones(len(beats), dtype=bool)
for candidate in early_candidates_set:
    if candidate in idx_to_beatpos:
        normal_mask[idx_to_beatpos[candidate]] = False
normal_template = np.median(beats[normal_mask], axis=0)

# %%
# -----------------------------------------------------------------------------
# 10. PVC DETECTION
# -----------------------------------------------------------------------------
# A beat is confirmed as PVC only if all three hold: early timing,
# compensatory pause (RR_pre + RR_post ~= 2x RR_median), and low
# correlation with the normal template (distorted morphology).
records = []
confirmed_pvcs = []
for idx in range(1, len(peaks) - 1):
    if idx not in idx_to_beatpos:
        continue

    beat_idx = idx_to_beatpos[idx]
    RR_pre = (peaks[idx] - peaks[idx - 1]) / fs
    RR_post = (peaks[idx + 1] - peaks[idx]) / fs
    RR_sum = RR_pre + RR_post
    is_compensatory = bool(np.abs(RR_sum - 2 * RR_median) < (0.2 * RR_median))

    corr = np.corrcoef(beats[beat_idx], normal_template)[0, 1]
    is_early = idx in early_candidates_set
    is_pvc = bool(is_early and corr < 0.75 and is_compensatory)

    if is_pvc:
        confirmed_pvcs.append(idx)

    print(f"Beat:{idx}, RR_pre:{RR_pre:.3f}, RR_post:{RR_post:.3f}, "
          f"Corr:{corr:.3f}, Compensatory:{is_compensatory}, PVC:{int(is_pvc)}")

    # Every beat (not just PVCs) is stored so the table can be used to
    # train a classifier later.
    records.append({
        'BEAT': idx,
        'FS': fs,
        'RR_PRE': RR_pre,
        'RR_POST': RR_post,
        'CORR': corr,
        'COMPENSATORY': is_compensatory,
        'PVC(1 FOR PVC, 0 FOR NORMAL)': int(is_pvc),
    })

print(f"Confirmed PVCs: {confirmed_pvcs}")

# %%
# -----------------------------------------------------------------------------
# 11. SAVING FEATURES TO CSV
# -----------------------------------------------------------------------------
# Consumed by the model training script.
nf = pd.DataFrame(records)
print(nf)
if not nf.empty:
    print(nf.head())
else:
    print("No early-beat candidates were found in this window.")

nf = pd.DataFrame(records)
nf.to_csv('ecg_features.csv', index=False)

# %%
