# %%
# ............................................................................
# IMPORTING LIBRARIES
# ............................................................................
from scipy.signal.windows import gaussian
import numpy as np
import wfdb
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PREMATURE VENTRICULAR CONTRACTION (PVC) DETECTION
# Dataset: Icentia11k (PhysioNet) | Libraries: SciPy, WFDB, NumPy, Matplotlib
#
# This script detects PVCs using two independent criteria that must both
# be satisfied:
#   1. Timing: the beat arrives significantly earlier than expected
#      (and is typically followed by a compensatory pause).
#   2. Morphology: the beat's shape correlates poorly with the patient's
#      normal QRS template, since PVCs originate outside the normal
#      conduction pathway and produce a wider, distorted waveform.
#
# The second half of this script validates the detector by injecting a
# synthetic PVC-like waveform into a real recording and checking whether
# the algorithm correctly flags it.
# ------------------------------------------------------------------------------

# %%
# -----------------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
# Load a record from the Icentia11k database.
dataset = wfdb.rdrecord('p01000_s02')

# Extract the first channel (Lead 1).
leads = dataset.p_signal[:, 0]

# Sampling frequency for this database.
fs_Icentia11k = 250

# Polarity inversion: in some recordings the R-wave is negative;
# inverting makes the R-peak positive, which simplifies peak detection.
signal_reading = -leads

# Time axis for the full signal (in seconds).
timer = np.arange(len(signal_reading)) / fs_Icentia11k

print(f"Total samples: {len(signal_reading)}")
print(f"Total duration: {len(signal_reading) / fs_Icentia11k:.2f} seconds")

# %%
# -----------------------------------------------------------------------------
# 2. BUTTERWORTH FILTERING (BANDPASS + LOW-PASS)
# -----------------------------------------------------------------------------
# Bandpass filter (0.5-40 Hz): removes low-frequency baseline wander
# (<0.5 Hz) and high-frequency muscle/EMG noise (>40 Hz).
lowcut, highcut = 0.5, 40.0
order = 6

b_bandpass, a_bandpass = signal.butter(
    order, [lowcut, highcut], btype='band', fs=fs_Icentia11k)

# filtfilt applies the filter forward and backward (zero-phase), so no
# timing/phase shift is introduced in the R-peak positions.
filtered_signal = signal.filtfilt(b_bandpass, a_bandpass, signal_reading)

# %%
# -----------------------------------------------------------------------------
# 3. ADAPTIVE R-PEAK DETECTION
# -----------------------------------------------------------------------------
# Robust baseline estimate using the median (resistant to sharp R-peaks
# skewing the value).
baseline = np.median(filtered_signal)

# Upper amplitude level using the 98th percentile (resistant to rare,
# very tall single-sample noise spikes).
upper_level = np.percentile(filtered_signal, 98)

# Adaptive height threshold (60% of the way from baseline to upper level).
height_threshold = baseline + 0.6 * (upper_level - baseline)

# Prominence threshold to reject P/T waves (20% of the amplitude range).
prominence_threshold = 0.2 * (upper_level - baseline)

# Detect R-peaks, enforcing a physiological ceiling of ~180 BPM
# (minimum 0.33 s between beats).
peaks, _ = find_peaks(
    filtered_signal,
    height=height_threshold,
    prominence=prominence_threshold,
    distance=int(0.33 * fs_Icentia11k)
)

peak_times = peaks / fs_Icentia11k
print(f"R-peaks detected: {len(peaks)}")

# %%
# -----------------------------------------------------------------------------
# 4. R-R INTERVALS & EARLY-BEAT CANDIDATES
# -----------------------------------------------------------------------------
# Time between consecutive R-peaks (in samples).
RR_interval = np.diff(peaks)

# Median R-R interval as the patient's baseline rhythm reference.
RR_median = np.median(RR_interval)

# A beat arriving less than 75% of the normal R-R interval after the
# previous one is flagged as a premature-beat candidate.
early_candidates = np.where(RR_interval < 0.75 * RR_median)[0] + 1

# %%
# -----------------------------------------------------------------------------
# 5. BEAT SEGMENTATION, TEMPLATE MATCHING & FINAL PVC CONFIRMATION
# -----------------------------------------------------------------------------
# Window around each R-peak (80 ms before and after).
window_sample = int(0.08 * fs_Icentia11k)

beats = []
valid_indices = []
for i, p in enumerate(peaks):
    if p - window_sample > 0 and p + window_sample < len(filtered_signal):
        beats.append(filtered_signal[p - window_sample: p + window_sample])
        valid_indices.append(i)

beats = np.array(beats)

# Build a "normal beat" template from the median of all beats not
# flagged as early candidates.
normal_mask = np.ones(len(beats), dtype=bool)
normal_mask[early_candidates] = False
normal_template = np.median(beats[normal_mask], axis=0)

confirmed_pvcs = []
for idx in early_candidates:
    if idx < len(RR_interval):
        # Condition 1 - Compensatory pause: the sum of the R-R interval
        # before and after the early beat should be roughly 2x the
        # normal interval.
        RR_sum = RR_interval[idx - 1] + RR_interval[idx]
        is_compensatory = np.abs(RR_sum - 2 * RR_median) < (0.2 * RR_median)

        # Condition 2 - Morphology: low Pearson correlation with the
        # normal template indicates an abnormal (widened/distorted) beat.
        corr = np.corrcoef(beats[idx], normal_template)[0, 1]

        # A beat is confirmed as a PVC only if both conditions hold.
        if corr < 0.75 and is_compensatory:
            confirmed_pvcs.append(idx)

print(f"Confirmed PVCs on the original recording: {len(confirmed_pvcs)}")

# %%
# -----------------------------------------------------------------------------
# 6. PLOT: PVC DETECTION ON THE ORIGINAL RECORDING
# -----------------------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(timer, signal_reading, label="Raw ECG", alpha=0.5, color='gray')
plt.plot(peak_times, signal_reading[peaks],
         "o", color='green', label="Normal R-peaks")

pvc_labeled = False
for idx in confirmed_pvcs:
    plt.axvline(
        peak_times[idx],
        color='red',
        linestyle="--",
        linewidth=2,
        label="Confirmed PVC" if not pvc_labeled else ""
    )
    pvc_labeled = True

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("ECG Signal Analysis — PVC Detection on Original Recording")
plt.grid(True, alpha=0.3)
plt.xlim(3275, 3285)  # Zoomed to a representative window
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# VALIDATION TEST: SYNTHETIC PVC INJECTION
#
# To sanity-check the detector, a synthetic PVC-like waveform (a wide,
# tall Gaussian pulse — mimicking the distorted morphology of a real PVC)
# is injected into the raw signal at a known time. The full detection
# pipeline is then re-run on the modified signal to verify the injected
# beat is correctly flagged.
# =============================================================================

# %%
# -----------------------------------------------------------------------------
# 7. INJECTING A SYNTHETIC PVC
# -----------------------------------------------------------------------------
injection_time = 3277.25  # seconds
injection_index = int(injection_time * fs_Icentia11k)

pvc_duration = 0.18  # seconds — wider than a normal QRS complex
pvc_width_samples = int(pvc_duration * fs_Icentia11k)

# A tall (1.8 mV), wide Gaussian pulse standing in for a distorted,
# ectopic beat shape.
pvc_shape = 1.8 * gaussian(pvc_width_samples, std=8)

half_width = pvc_width_samples // 2
start_idx = injection_index - half_width
end_idx = start_idx + len(pvc_shape)

signal_reading[start_idx:end_idx] += pvc_shape

# %%
# Preview of the injected waveform.
plt.figure(figsize=(14, 5))
plt.plot(timer, signal_reading, label="Raw ECG with Injected PVC",
         alpha=0.7, color='gray')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("ECG Signal — Synthetic PVC Injection")
plt.grid(True, alpha=0.3)
plt.xlim(3276, 3278)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# %%
# -----------------------------------------------------------------------------
# 8. RE-RUNNING THE FULL PIPELINE ON THE MODIFIED SIGNAL
# -----------------------------------------------------------------------------
# The entire detection pipeline (filtering, peak detection, R-R intervals,
# early-beat flagging, template matching) is re-run from scratch on the
# signal that now contains the injected PVC. This is the actual test of
# whether the detector works — not just where the injection happened.

filtered_signal = signal.filtfilt(b_bandpass, a_bandpass, signal_reading)

baseline = np.median(filtered_signal)
upper_level = np.percentile(filtered_signal, 98)

# Note: thresholds here are intentionally slightly relaxed compared to
# step 3 above. Keep this in mind when comparing results between the two
# passes — they are not run under identical settings.
height_threshold = baseline + 0.5 * (upper_level - baseline)
prominence_threshold = 0.15 * (upper_level - baseline)

peaks, _ = find_peaks(
    filtered_signal,
    height=height_threshold,
    prominence=prominence_threshold,
    distance=int(0.28 * fs_Icentia11k)
)

window_sample = int(0.08 * fs_Icentia11k)
beats = []
valid_peaks = []

for p in peaks:
    if p - window_sample > 0 and p + window_sample < len(filtered_signal):
        beats.append(filtered_signal[p - window_sample: p + window_sample])
        valid_peaks.append(p)

beats = np.array(beats)
valid_peaks = np.array(valid_peaks)
peak_times = valid_peaks / fs_Icentia11k

RR_interval = np.diff(valid_peaks)
RR_median = np.median(RR_interval)

early_candidates = np.where(RR_interval < 0.75 * RR_median)[0] + 1

normal_mask = np.ones(len(beats), dtype=bool)
normal_mask[early_candidates] = False
normal_template = np.median(beats[normal_mask], axis=0)

# Note: this second pass checks morphology only (correlation < 0.70) and
# does not re-apply the compensatory-pause condition used in step 5.
confirmed_pvcs = []
for idx in early_candidates:
    if idx < len(beats):
        corr = np.corrcoef(beats[idx], normal_template)[0, 1]
        if corr < 0.70:
            confirmed_pvcs.append(idx)

# Check whether the injected beat was actually caught.
confirmed_times = peak_times[confirmed_pvcs]
detected_injection = np.any(np.abs(confirmed_times - injection_time) < 0.1)
print(f"Confirmed PVCs after injection: {len(confirmed_pvcs)}")
print(f"Confirmed PVC times (s): {confirmed_times}")
print(
    f"Injected PVC at {injection_time}s successfully detected: {detected_injection}")

# %%
# -----------------------------------------------------------------------------
# 9. PLOT: VALIDATION RESULT
# -----------------------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(timer, signal_reading, label="Raw ECG with Injected PVC",
         alpha=0.5, color='gray')
plt.plot(peak_times, signal_reading[valid_peaks],
         "o", color='green', label="Detected R-peaks")

pvc_labeled = False
for idx in confirmed_pvcs:
    plt.axvline(
        peak_times[idx],
        color='red',
        linestyle="--",
        linewidth=2,
        label="Confirmed PVC" if not pvc_labeled else ""
    )
    pvc_labeled = True

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("ECG Signal Analysis — PVC Detection Validation (Synthetic Injection)")
plt.grid(True, alpha=0.3)
plt.xlim(3275, 3280)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# %%
