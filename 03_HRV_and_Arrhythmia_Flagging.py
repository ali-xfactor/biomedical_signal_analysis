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
# Reading record '122' from the MIT-BIH Arrhythmia Database (PhysioNet).
data_set = wfdb.rdrecord('122')

# The record contains multiple leads (channels); we select the first one (Lead 0).
sparate_leads = data_set.p_signal[:, 0]

# Sampling frequency of this database: 360 samples per second.
fs = 360

# Selecting a 5-second window to analyze (0 to 5 seconds).
start_timer = 0
end_timer = 5 * fs
t = np.arange(start_timer, end_timer) / fs

# Slicing out the ECG signal for this time window.
ecg_signal = sparate_leads[start_timer:end_timer]

# %%
# Plotting the raw ECG signal in the time domain, before any processing.
plt.figure(figsize=(10, 4))
plt.plot(t, ecg_signal, color='gray')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("time")
plt.ylabel("millivolt")
plt.title("Raw ECG Signal (Time Domain)")
plt.show()

# %%
# -----------------------------------------------------------------------------
# 2. DC OFFSET REMOVAL & FREQUENCY SPECTRUM (BASELINE CHECK)
# -----------------------------------------------------------------------------
# Removing the DC offset (average baseline voltage) before running FFT.
# Without this step, a huge artificial peak would appear at 0 Hz and hide
# the real frequency content we care about.
N = end_timer - start_timer
mean_val = ecg_signal - np.mean(ecg_signal)

# Computing the FFT to see which frequencies are present in the clean signal.
fft_signal = fft(mean_val)
frequans = fftfreq(len(ecg_signal), d=1 / fs)
freqs_positive = frequans[:N // 2]
magnitude = np.abs(fft_signal)[:N // 2]

# %%
# Plotting the frequency spectrum of the original (not-yet-noisy) signal.
# This is our baseline reference — useful to compare against later once we
# add noise and then filter it back out.
plt.figure(figsize=(10, 4))
plt.plot(freqs_positive, magnitude, color='red')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 60)
plt.xlabel("Frequency(Hz)", fontsize=11)
plt.ylabel("Magnitude", fontsize=11)
plt.title("(ECG Spectrum Analysis)", fontsize=12)
plt.show()

# %%
# -----------------------------------------------------------------------------
# 3. SIMULATING POWERLINE NOISE (50 Hz)
# -----------------------------------------------------------------------------
# To test our filters, we deliberately contaminate the clean signal with a
# synthetic 50 Hz sine wave — simulating real-world electrical interference
# from the power grid, which is a very common artifact in ECG recordings.
noise_amplitude = 0.1
noise_50hz = noise_amplitude * np.sin(2 * np.pi * 50.0 * t)
noisy_sig = ecg_signal + noise_50hz

# %%
# Plotting the clean signal against the noisy version, so the added
# interference is visible before we try to remove it.
plt.figure(figsize=(12, 5))
plt.plot(t, ecg_signal, label="Normal Signal", color='red', alpha=0.5)
plt.plot(t, noisy_sig, color='gray',
         label="Noisy Signal (50 Hz added)", alpha=0.6)
plt.legend()
plt.title("signal with 50hz")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# %%
# Computing the FFT of the noisy signal — we expect to see a new, sharp
# spike at exactly 50 Hz that wasn't present in the original spectrum.
signal_centured = noisy_sig - np.mean(noisy_sig)
fft_noisy = fft(signal_centured)
frequans_noisy = fftfreq(len(noisy_sig), d=1 / fs)
noise_positive = frequans_noisy[:N // 2]
noise_magnitude = np.abs(fft_noisy)[:N // 2]

# %%
# Plotting the noisy spectrum — confirms the 50 Hz contamination visually.
plt.figure(figsize=(10, 4))
plt.plot(noise_positive, noise_magnitude, color='gray')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 60)
plt.xlabel("Frequency(Hz)", fontsize=11)
plt.ylabel("Magnitude", fontsize=11)
plt.title("50_hz noise", fontsize=12)
plt.show()

# %%
# -----------------------------------------------------------------------------
# 4. NOTCH FILTER: REMOVING THE 50 Hz POWERLINE NOISE
# -----------------------------------------------------------------------------
# A notch filter surgically removes a single, narrow frequency band while
# leaving everything else untouched — ideal for killing powerline noise
# without distorting the rest of the ECG waveform.
notch_freq = 50.0     # The exact frequency we want to eliminate.
quality_factor = 30.0  # Controls how narrow/sharp the notch is.

b_notch, a_notch = signal.iirnotch(w0=notch_freq, Q=quality_factor, fs=fs)
print("Filter numerator coefficients (b):", b_notch)
print("Filter denominator coefficients (a):", a_notch)

# filtfilt applies the filter forward and backward, which cancels out any
# phase distortion (zero-phase filtering) — critical for medical signals,
# since we don't want to shift the timing of the heartbeats.
filtered_data = signal.filtfilt(b_notch, a_notch, noisy_sig)
signal_notch = filtered_data - np.mean(filtered_data)
fft_notch = fft(signal_notch)
frequans_notch = fftfreq(len(signal_notch), d=1 / fs)
notch_positive = frequans_notch[:N // 2]
notch_magnitude = np.abs(fft_notch)[:N // 2]

# %%
# Plotting the spectrum after the notch filter — the 50 Hz spike should
# now be gone (or drastically reduced) compared to the noisy spectrum above.
plt.figure(figsize=(10, 4))
plt.plot(notch_positive, notch_magnitude, color='gray')
plt.xlim(0, 60)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("Spectrum After Notch Filter (50 Hz removed)")
plt.show()

# %%
# -----------------------------------------------------------------------------
# 5. BUTTERWORTH LOW-PASS FILTER: REMOVING HIGH-FREQUENCY NOISE
# -----------------------------------------------------------------------------
# After the notch filter handles the powerline noise, a low-pass Butterworth
# filter cleans up remaining high-frequency artifacts (e.g. muscle noise),
# since the clinically useful content of an ECG signal sits below ~30-40 Hz.
fc = 30.0  # Cutoff frequency in Hz — frequencies above this are attenuated.
N = 4      # Filter order — higher order = steeper roll-off at the cutoff.

b_butter, a_buuter = signal.butter(N, fc, btype='low', fs=fs)
filtered_butter = signal.filtfilt(b_butter, a_buuter, noisy_sig)

N = len(filtered_butter)
fft_butter = fft(filtered_butter - np.mean(filtered_butter))
frequans_butter = fftfreq(len(filtered_butter), d=1 / fs)
butter_positive = frequans_butter[:N // 2]
butter_magnitude = np.abs(fft_butter)[:N // 2]
# %%
# Plotting the noisy signal vs. the fully filtered signal — this is the
# "before and after" comparison that shows the filtering pipeline works.
plt.figure(figsize=(12, 5))
plt.plot(t, noisy_sig, label="Noisy ECG", alpha=0.5)
plt.plot(t, filtered_butter, label="Butterworth 30 Hz")
plt.xlim(0, 5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("ECG Before and After Butterworth Filter")
plt.grid(True)
plt.legend()
plt.show()

# Plotting the final, clean frequency spectrum — should show the ECG's
# natural frequency content with no lingering 50 Hz spike and no
# high-frequency clutter above the 30 Hz cutoff.
plt.figure(figsize=(12, 5))
plt.plot(butter_positive, butter_magnitude, alpha=0.5)
plt.xlim(0, 60)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("Final Spectrum After Butterworth Filtering")
plt.show()
# %%
# -----------------------------------------------------------------------------
# 6. R-PEAK DETECTION
# -----------------------------------------------------------------------------
# Locating each heartbeat's R-peak (the tall, sharp spike in the QRS complex)
# on the fully filtered signal.
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
baseline = np.median(filtered_butter)

# 2. Determine the upper amplitude level using the 98th percentile
#    (ignores single massive artifacts that max() would catch)
upper_level = np.percentile(filtered_butter, 98)

# 3. Adaptive threshold
# We use 60% of the distance between baseline and upper_level.
# This is safer than using exactly 'upper_level', which might miss slightly shorter R-peaks.
height_threshold = baseline + 0.6 * (upper_level - baseline)

# 4. Adaptive prominence to reject small P/T waves (20% of the peak-to-baseline difference)
prominence_threshold = 0.2 * (upper_level - baseline)

peaks_filtered, properties = find_peaks(
    filtered_butter,
    height=height_threshold,
    prominence=prominence_threshold,
    distance=int(0.33 * fs)
)
print(f"number of R-peaks: {len(peaks_filtered)}")
print(f"R-peaks {peaks_filtered}")
# %%
# -----------------------------------------------------------------------------
# 7. HEART RATE (BPM) & ARRHYTHMIA FLAGGING
# -----------------------------------------------------------------------------
# Calculating the time interval between consecutive R-peaks (R-R intervals).
RR_interval = np.diff(peaks_filtered) / fs
print(
    f"The interval between each heartbeat per second(RR-interval):{RR_interval}")
# Converting each R-R interval into an instantaneous heart rate (BPM).
Bpm = 60 / RR_interval
print(f"Instantaneous heart rate:{Bpm}")
print(f"Average heart rate:{np.mean(Bpm):.2f} BPM")
# Simple rule-based flagging based on average BPM.
# Note: these are general clinical thresholds for resting heart rate —
# they classify the recorded signal, not a live diagnosis of any person.
if np.mean(Bpm) < 60:
    print("Patient's condition in a state of bradycardia")
elif np.mean(Bpm) > 100:
    print("Patient's condition in a state of tachycardia")
else:
    print("Patient's condition in a state of normal heart beat")
# %%
# -----------------------------------------------------------------------------
# 8. HEART RATE VARIABILITY (HRV) — TIME-DOMAIN METRICS
# -----------------------------------------------------------------------------
# These three metrics are the most common time-domain HRV indicators, used
# to assess autonomic nervous system activity (sympathetic vs. parasympathetic
# balance) from the pattern of R-R intervals.

# --- 1. SDNN: Standard Deviation of NN (R-R) intervals ---
# Reflects OVERALL heart rate variability across the whole recording.
SDNN = np.std(RR_interval, ddof=1)
print(f"SDNN:{SDNN * 1000:.2f} ms")

# --- 2. RMSSD: Root Mean Square of Successive Differences ---
# Reflects SHORT-TERM, beat-to-beat variability — a strong indicator of
# parasympathetic (vagal/"rest and recover") nervous system activity.
#
# IMPORTANT: the difference must be taken BEFORE squaring
# (i.e. (RR[i+1] - RR[i])**2 ), not after squaring each RR value first.
# Squaring first and then differencing computes RR[i+1]**2 - RR[i]**2,
# which is mathematically a different (and incorrect) quantity.
RMSSD = np.sqrt(
    np.mean(
        np.diff(RR_interval) ** 2
    )
)
# RR_interval is in seconds, so multiply by 1000 (not 100) to convert to ms.
print(f"RMSSD:{RMSSD * 1000:.2f} ms")

# --- 3. pNN50: Percentage of successive RR differences greater than 50 ms ---
# Another parasympathetic-activity indicator, more intuitive for clinicians
# than RMSSD since it's expressed as a simple percentage.
RR_diff = np.abs(np.diff(RR_interval))
pNN50 = np.mean(RR_diff > 0.05) * 100  # 0.05 s = 50 ms, since RR is in seconds
print(f"pNN50:{pNN50:.2f}%")
# %%
# -----------------------------------------------------------------------------
# 9. VISUALIZING DETECTED R-PEAKS & HRV METRICS
# -----------------------------------------------------------------------------
# Final sanity check: plotting the filtered signal with every detected
# R-peak marked, and displaying the clinical HRV metrics directly on the plot.
plt.figure(figsize=(10, 4))
plt.plot(
    filtered_butter,
    label='Filtered ECG Signal',
    color='black',
    alpha=0.7
)

plt.plot(
    peaks_filtered,
    filtered_butter[peaks_filtered],
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

hrv_text = (
    f"Heart Rate: {np.mean(Bpm):.1f} BPM\n"
    f"SDNN: {SDNN * 1000:.1f} ms\n"
    f"RMSSD: {RMSSD * 1000:.1f} ms\n"
    f"pNN50: {pNN50:.1f} %"
)

bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='gray', alpha=0.9)
plt.text(
    0.01, 0.04,
    hrv_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='bottom',
    bbox=bbox_props
)

plt.title("ECG Analysis: R-Peak Detection & HRV Metrics")
plt.xlabel("(Sample)")
plt.ylabel("(Amplitude (mV))")
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
# %%
