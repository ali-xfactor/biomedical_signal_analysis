# %%
# -----------------------------------------------------------------------------
# IMPORTING LIBRARIES
# -----------------------------------------------------------------------------
import numpy as np
import wfdb
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# %%
# -----------------------------------------------------------------------------
# 1. LOADING THE CLINICAL DATASET
# -----------------------------------------------------------------------------
# Reading record '117' from the MIT-BIH Arrhythmia Database (PhysioNet).
# This record was collected from a 69-year-old male patient, sampled at
# 360 Hz across two leads (MLII and V2). We work with the first lead (MLII),
# the standard modified limb lead used in most rhythm analysis.
record = wfdb.rdrecord('103')
ecg_raw = record.p_signal[:, 0]

# Sampling frequency: the device captured 360 data points per second.
fs = 360

# Selecting a 5-second analysis window (0 to 5 seconds).
start = 0
end = 5 * fs
N = end - start
t = np.arange(start, end) / fs

ecg_signal = ecg_raw[start:end]

# %%
# Plotting the raw ECG signal in the time domain, before any processing.
# This is the unfiltered baseline waveform as recorded by the device.
plt.figure(figsize=(10, 4))
plt.plot(t, ecg_signal, color='gray')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("Raw ECG Signal — Record 117, Lead MLII (Time Domain)")
plt.show()

# %%
# -----------------------------------------------------------------------------
# 2. DC OFFSET REMOVAL
# -----------------------------------------------------------------------------
# Biomedical signals recorded by clinical hardware typically ride on a
# non-zero baseline voltage (DC offset), caused by electrode-skin contact
# and amplifier characteristics. If left uncorrected, this offset produces
# an artificially large spike at 0 Hz in the frequency domain, which would
# dwarf every physiologically meaningful frequency component and make the
# spectrum unreadable. Subtracting the window's mean centers the signal
# around zero and eliminates this artifact.
mean_val = np.mean(ecg_signal)
ecg_centered = ecg_signal - mean_val

# %%
# -----------------------------------------------------------------------------
# 3. BASELINE FREQUENCY SPECTRUM (FFT)
# -----------------------------------------------------------------------------
# Computing the FFT to characterize the frequency content of the clean
# signal before any noise is introduced — this serves as our reference
# spectrum for later comparison.
fft_signal = fft(ecg_centered)
freqs = fftfreq(N, d=1 / fs)

# For a real-valued signal, the FFT output is symmetric: the second half is
# a mirror of the first. We therefore only need the positive-frequency half
# (up to the Nyquist frequency, fs/2 = 180 Hz) to fully describe the signal.
freqs_positive = freqs[:N // 2]
magnitude = np.abs(fft_signal)[:N // 2]

# %%
plt.figure(figsize=(10, 4))
plt.plot(freqs_positive, magnitude, color='navy')
plt.xlim(0, 60)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Frequency (Hz)", fontsize=11)
plt.ylabel("Magnitude", fontsize=11)
plt.title("Baseline Frequency Spectrum — Record 117", fontsize=12)
plt.show()

# Clinical interpretation of the ECG frequency spectrum:
#   0.5 - 4 Hz  : T-wave morphology and slow baseline wander
#   8 - 20 Hz   : the QRS complex — the sharpest, most energetic feature
#                 of a normal heartbeat
#   50 / 60 Hz  : powerline interference, not physiological — a sharp
#                 spike here indicates electrical noise contamination

# %%
# -----------------------------------------------------------------------------
# 4. SIMULATING POWERLINE INTERFERENCE (50 Hz)
# -----------------------------------------------------------------------------
# To validate that our filtering pipeline actually works, we deliberately
# inject a synthetic 50 Hz sine wave into the clean signal, replicating the
# electrical interference commonly picked up from AC power lines during
# real-world ECG recording.
noise_amplitude = 0.1
noise_50hz = noise_amplitude * np.sin(2 * np.pi * 50.0 * t)
noisy_signal = ecg_signal + noise_50hz

# %%
plt.figure(figsize=(12, 5))
plt.plot(t, ecg_signal, label="Original Signal", color='red', alpha=0.5)
plt.plot(t, noisy_signal, color='gray',
         label="Noisy Signal (50 Hz Interference Added)", alpha=0.6)
plt.legend()
plt.title("Record 117 — Before and After Synthetic Noise Injection")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# %%
# Confirming the contamination in the frequency domain: a new, sharp spike
# should now appear at exactly 50 Hz that was not present in the baseline
# spectrum above.
noisy_centered = noisy_signal - np.mean(noisy_signal)
fft_noisy = fft(noisy_centered)
freqs_noisy = fftfreq(N, d=1 / fs)
noisy_freqs_positive = freqs_noisy[:N // 2]
noisy_magnitude = np.abs(fft_noisy)[:N // 2]

plt.figure(figsize=(10, 4))
plt.plot(noisy_freqs_positive, noisy_magnitude, color='gray')
plt.xlim(0, 60)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Frequency (Hz)", fontsize=11)
plt.ylabel("Magnitude", fontsize=11)
plt.title("Spectrum After Noise Injection — 50 Hz Contamination Visible", fontsize=12)
plt.show()

# %%
# -----------------------------------------------------------------------------
# 5. NOTCH FILTER: REMOVING THE 50 Hz POWERLINE NOISE
# -----------------------------------------------------------------------------
# A notch filter is a narrow band-stop filter — it surgically removes a
# single target frequency while leaving all other frequencies untouched.
# This makes it the ideal tool for eliminating powerline interference
# without distorting the rest of the ECG waveform.
notch_freq = 50.0       # The exact frequency to eliminate.
quality_factor = 30.0   # Controls the notch's sharpness: bandwidth = f0 / Q.
# A higher Q means a narrower, more surgical cut.

b_notch, a_notch = signal.iirnotch(w0=notch_freq, Q=quality_factor, fs=fs)
print("Notch filter numerator coefficients (b):", b_notch)
print("Notch filter denominator coefficients (a):", a_notch)

# filtfilt applies the filter both forward and backward through the signal.
# This zero-phase technique cancels out the phase distortion (time delay)
# that a single-pass filter would introduce — essential here, since we
# cannot afford to shift the timing of the heartbeats we are about to detect.
after_notch = signal.filtfilt(b_notch, a_notch, noisy_signal)

# %%
# Verifying the notch filter's effect in the frequency domain — the 50 Hz
# spike should now be gone.
notch_centered = after_notch - np.mean(after_notch)
fft_notch = fft(notch_centered)
freqs_notch = fftfreq(N, d=1 / fs)
notch_freqs_positive = freqs_notch[:N // 2]
notch_magnitude = np.abs(fft_notch)[:N // 2]

plt.figure(figsize=(10, 4))
plt.plot(notch_freqs_positive, notch_magnitude, color='purple')
plt.xlim(0, 60)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("Spectrum After Notch Filtering — 50 Hz Component Removed")
plt.show()

# %%
# -----------------------------------------------------------------------------
# 6. BUTTERWORTH LOW-PASS FILTER: REMOVING RESIDUAL HIGH-FREQUENCY NOISE
# -----------------------------------------------------------------------------
# The Butterworth filter is known as a "maximally flat" filter: within its
# passband, it introduces no ripple or distortion to the frequencies it
# lets through — a critical property for medical signals, where preserving
# the true shape of the waveform (e.g. the QRS complex) matters as much as
# removing noise.
#
# Filter order (N): controls how sharply the filter cuts off at the
# boundary frequency. A low order (e.g. N=1) rolls off gently, allowing
# some noise above the cutoff to leak through; a higher order (e.g. N=4)
# cuts off steeply. For biomedical applications, N=2 to N=4 is standard —
# orders much higher than this can introduce numerical instability.
#
# Cutoff frequency (fc): clinically, 30-40 Hz is the conventional cutoff
# for ECG signals, since it removes high-frequency muscle (EMG) artifacts
# while preserving virtually all diagnostically relevant content of the
# QRS complex.
fc = 30.0
order = 4

b_butter, a_butter = signal.butter(order, fc, btype='low', fs=fs)
filtered_signal = signal.filtfilt(b_butter, a_butter, after_notch)

# %%
# Comparing the noisy input against the fully filtered output — this is
# the definitive "before and after" view of the entire filtering pipeline.
plt.figure(figsize=(12, 5))
plt.plot(t, noisy_signal, label="Noisy ECG (Pre-filtering)", alpha=0.5)
plt.plot(t, filtered_signal, label="Fully Filtered ECG (Notch + Butterworth)",
         color='darkgreen')
plt.xlim(0, 5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("Record 117 — ECG Before and After Full Filtering Pipeline")
plt.grid(True)
plt.legend()
plt.show()

# %%
# Final frequency spectrum: should show the ECG's natural frequency content
# with no lingering 50 Hz spike and no high-frequency clutter above 30 Hz.
N_final = len(filtered_signal)
fft_final = fft(filtered_signal - np.mean(filtered_signal))
freqs_final = fftfreq(N_final, d=1 / fs)
final_freqs_positive = freqs_final[:N_final // 2]
final_magnitude = np.abs(fft_final)[:N_final // 2]

plt.figure(figsize=(12, 5))
plt.plot(final_freqs_positive, final_magnitude, color='darkgreen', alpha=0.7)
plt.xlim(0, 60)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("Final Spectrum After Complete Filtering Pipeline")
plt.show()

# %%
# -----------------------------------------------------------------------------
# 7. R-PEAK DETECTION
# -----------------------------------------------------------------------------
# Locating each heartbeat's R-peak — the tall, sharp spike at the center
# of the QRS complex — on the fully filtered signal.
#
# Parameters for find_peaks:
#   height     : Adaptive minimum amplitude a point must reach to count as a
#                real R-peak. Determined from the 98th percentile of the signal
#                to adapt to each specific ECG record's voltage range.
#   prominence : Minimum vertical distance a peak must stand out from its
#                surrounding baseline. Effectively filters out smaller P and T waves.
#   distance   : Minimum sample gap between two peaks (0.33 * fs). This is
#                based on the physiological ceiling of ~180 BPM for a healthy
#                heart (1 beat approx every 0.33 seconds).

# 1. Estimate the baseline robustly using the median (ignores extreme spikes)
baseline = np.median(filtered_signal)

# 2. Determine the upper amplitude level using the 98th percentile
#    (ignores single massive artifacts that max() would catch)
upper_level = np.percentile(filtered_signal, 98)

# 3. Adaptive threshold
# We use 60% of the distance between baseline and upper_level.
# This is safer than using exactly 'upper_level', which might miss slightly shorter R-peaks.
height_threshold = baseline + 0.6 * (upper_level - baseline)

# 4. Adaptive prominence to reject small P/T waves (20% of the peak-to-baseline difference)
prominence_threshold = 0.2 * (upper_level - baseline)

# 5. Detect Peaks
peaks, properties = find_peaks(
    filtered_signal,
    height=height_threshold,
    prominence=prominence_threshold,
    distance=int(0.33 * fs)
)

print(f"Number of R-peaks detected: {len(peaks)}")
print(f"R-peak sample indices: {peaks}")

# 6. Visualization
plt.figure(figsize=(10, 4))
plt.plot(
    filtered_signal,
    label='Filtered ECG Signal',
    color='black',
    alpha=0.7
)

plt.plot(
    peaks,
    filtered_signal[peaks],
    "x",
    color='red',
    markersize=10,
    markeredgewidth=3,
    label='Detected R-peaks'
)

plt.axhline(
    y=height_threshold,
    color='green',
    linestyle='--',
    linewidth=1.5,
    label=f'Adaptive Threshold ({height_threshold:.2f} mV)'
)

plt.title("R-Peak Detection on Filtered Signal — Record 103")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (mV)")
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
# %%
# -----------------------------------------------------------------------------
# 8. HEART RATE (BPM)
# -----------------------------------------------------------------------------
# Calculating the R-R intervals (the time gap between consecutive R-peaks)
# and converting them into an instantaneous and average heart rate.
RR_intervals = np.diff(peaks) / fs
print(f"R-R intervals (seconds): {RR_intervals}")

bpm = 60 / RR_intervals
print(f"Instantaneous heart rate (BPM): {np.round(bpm, 2)}")
print(f"Average heart rate: {np.mean(bpm):.2f} BPM")

# %%
