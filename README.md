# Robust Algorithmic Pipeline for Clinical ECG Processing and Autonomic Nervous System Evaluation

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/PhysioNet-MIT--BIH-red?style=for-the-badge" alt="MIT-BIH">
</div>

## 📑 Abstract
This repository presents a comprehensive, end-to-end biomedical digital signal processing (DSP) pipeline designed for the extraction, conditioning, and clinical analysis of Electrocardiogram (ECG) data. By mitigating electrical and physiological artifacts using advanced adaptive filtering techniques, the system accurately extracts R-peaks to perform time-domain Heart Rate Variability (HRV) analysis. This project serves as a foundational computational module for integration into bio-robotic monitoring systems and automated diagnostic software.

---

## 🔬 Methodology & Architecture

The signal processing architecture is designed to handle raw, noisy physiological data and transform it into actionable clinical metrics. 

### 1. Signal Acquisition & Preprocessing
Data is sourced directly from the **MIT-BIH Arrhythmia Database** (sampled at 360 Hz). The initial phase involves isolating the signal from baseline wander (DC offset) to prepare for precise frequency-domain analysis.

### 2. Spectral Analysis & Adaptive Filtering
To ensure diagnostic integrity, the pipeline employs specialized filtering to surgically eliminate noise without altering the critical QRS morphology:
* **Zero-Phase IIR Notch Filter:** Targets and attenuates 50 Hz powerline interference ($Q=30$).
* **4th-Order Butterworth Low-Pass Filter:** Suppresses high-frequency physiological artifacts (e.g., EMG noise) with a cutoff frequency ($f_c$) of 30 Hz.

<div align="center">
  <img src="50hz.noise.png" alt="50 Hz Powerline Interference Mitigation" width="700">
  <br>
  <em>Figure 1: Spectral analysis (FFT) demonstrating the isolation and attenuation of 50 Hz powerline interference using a zero-phase Notch filter.</em>
  
  <br><br>

  <img src="butruth signal.png" alt="Butterworth Low-Pass Filtering" width="700">
  <br>
  <em>Figure 2: Signal conditioning utilizing a 4th-Order Butterworth Low-Pass filter to suppress high-frequency physiological artifacts (e.g., EMG noise) while preserving critical QRS morphology.</em>
</div>

### 3. Algorithmic Feature Extraction
A dynamic, rule-based algorithm applies adaptive thresholding—calculated via statistical medians and 98th percentiles—to accurately detect ventricular depolarizations (R-peaks). The algorithm is highly resistant to T-wave interference and fluctuating voltage amplitudes.

<div align="center">
  <img src="ecg-result.png" alt="Adaptive R-Peak Detection" width="800">
  <br>
  <em>Figure 2: Filtered ECG signal showcasing automated R-peak detection (red markers) utilizing dynamic thresholding (green dashed line).</em>
</div>

---

## 📊 Clinical Diagnostics & HRV Analysis

Beyond raw signal processing, the pipeline computes critical physiological parameters to evaluate the Autonomic Nervous System (ANS) and flag potential arrhythmias (Bradycardia & Tachycardia).

### Extracted HRV Metrics (Time-Domain):
* **SDNN (Standard Deviation of NN intervals):** Quantifies overall heart rate variability and long-term autonomic regulation.
* **RMSSD (Root Mean Square of Successive Differences):** Serves as a primary indicator of parasympathetic (vagal) tone.
* **pNN50:** Evaluates the percentage of successive R-R interval differences exceeding 50 milliseconds.

<div align="center">
  <img src="r-peak-hrv.png" alt="HRV Analysis Output" width="700">
  <br>
  <em>Figure 3: Heart Rate Variability (HRV) metrics extraction and arrhythmia diagnostic flagging.</em>
</div>

---

## Applications in Telemedicine & Home Medical Robotics

The algorithms developed in this repository are optimized for low-latency physiological monitoring. The methodology lays the groundwork for real-time cardiac assessment modules in **Home Medical Robotics** and automated patient-care systems, where robust, autonomous signal interpretation is critical for remote diagnostics.

---

## 💻 Technical Stack & Deployment

* **Language:** Python 3.x
* **Core Libraries:** 'numpy' (matrix operations), 'scipy' (FFT, signal processing, adaptive peak detection), 'matplotlib' (clinical rendering), 'wfdb' (PhysioNet data parsing).

### Installation & Execution

# 1. Clone the repository:
git clone: [https://github.com/ali-xfctor/biomedical_signal_analysis.git](https://github.com/ali-xfactor/biomedical_signal_analysis)

# 2. Install dependencies:
pip install numpy scipy matplotlib wfdb

# 3. Download Dataset (e.g., Record 100 or 122) from PhysioNet and place in the root directory.

# 4. Execute the analysis pipeline:
     python ecg_analysis.py

---

## 👤 Author
**Ali Farhadi**

**Undergraduate Researcher | Biomedical Engineering**

**Focus: Biomedical Signal Processing & Autonomous Medical Robotics**

## 📚 Acknowledgments & Dataset Attribution
This research relies on the **MIT-BIH Arrhythmia Database**, openly provided by **PhysioNet**. 

Special thanks to the open-source community for maintaining the clinical data structures essential for engineering advancements.
