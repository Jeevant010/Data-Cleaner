# Complete Mathematics of Audio Trimming and Preprocessing

This document outlines the rigorous mathematical foundation behind the audio trimming and preprocessing algorithms used in the `Data-Cleaner` pipeline. It covers digital audio representation, signal detection, and quality gating from first principles.

---

## 1. Digital Audio Fundamentals

### 1.1 Analog to Digital Conversion (Nyquist-Shannon)
An analog sound wave is a continuous function of time, $x(t)$. To process this on a computer, it is sampled at a discrete sampling rate, $f_s$ (measured in Hertz). 

According to the **Nyquist-Shannon Sampling Theorem**, to perfectly reconstruct a signal with maximum frequency $f_{max}$, the sampling rate must be:
$$f_s \ge 2 \cdot f_{max}$$

In this project, the chosen sampling rate is $f_s = 22,050 \text{ Hz}$. 
The discrete signal is represented as an array of amplitudes:
$$y[n] = x(n \cdot T_s)$$
where $T_s = \frac{1}{f_s}$ is the sampling period, and $n \in \{0, 1, 2, \dots, N-1\}$.

### 1.2 Clip Duration and Sample Length
When configuring the trimmer to extract a clip of duration $T_{target}$ (e.g., $750\text{ ms} = 0.75\text{ s}$), the exact number of discrete samples $N$ required is:
$$N = \lfloor f_s \cdot T_{target} \rfloor$$
For $750\text{ ms}$ at $22,050\text{ Hz}$:
$$N = \lfloor 22,050 \cdot 0.75 \rfloor = 16,537 \text{ samples}$$

---

## 2. Onset Detection Mathematics (Gunshot Trimming)

The gunshot trimmer relies on finding the exact moment a loud impulse begins (the onset). It uses three concurrent mathematical methods, ranking the candidates.

### 2.1 Method 1: Spectral Flux (Librosa Onset Detect)
This method detects sudden changes in the frequency domain. 

**Step 1: Short-Time Fourier Transform (STFT)**
The continuous signal is broken into short, overlapping windows using a window function $w[m]$ (e.g., Hann window). The discrete STFT is given by:
$$X(k, m) = \sum_{n=0}^{N_{fft}-1} y[n + m \cdot H] \cdot w[n] \cdot e^{-j \frac{2\pi}{N_{fft}} k n}$$
Where:
- $X(k, m)$ is the complex spectrum at frequency bin $k$ and time frame $m$.
- $H$ is the hop length (typically 128 or 512 samples).
- $N_{fft}$ is the FFT window size.
- $w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N_{fft} - 1}\right)\right)$ is the Hann window.

**Step 2: Magnitude Spectrogram**
The energy (magnitude) of the spectrum is:
$$S(k, m) = |X(k, m)|$$

**Step 3: Spectral Flux (Onset Envelope)**
Spectral flux measures the positive difference in magnitude between consecutive time frames across all frequencies. The onset strength $O(m)$ at frame $m$ is:
$$O(m) = \sum_{k} HWF(S(k, m) - S(k, m-1))$$
Where $HWF(z) = \max(0, z)$ is the Half-Wave Rectifier, ensuring we only measure increases in energy (onsets), not decreases (offsets).

Peaks in the sequence $O(m)$ are detected using a local maximum filter, yielding the onset frames $m_{onset}$. The physical sample index is calculated as:
$$n_{onset} = m_{onset} \cdot H$$

### 2.2 Method 2: Raw Amplitude Peak
The simplest physical indicator of a gunshot is the absolute maximum amplitude of the waveform.
$$n_{peak} = \arg\max_{n} |y[n]|$$

### 2.3 Method 3: Smoothed Energy Envelope
To prevent single-sample anomalies from tricking the detector, we calculate a rolling average of the absolute amplitude using 1D Discrete Convolution.

Given a smoothing kernel $k$ of size $K = \max(3, \lfloor 0.005 \cdot f_s \rfloor)$:
$$k[i] = \frac{1}{K} \text{ for } i \in \{0, 1, \dots, K-1\}$$

The smoothed envelope $E[n]$ is the convolution of the absolute signal with the kernel:
$$E[n] = (|y| * k)[n] = \sum_{i=0}^{K-1} |y[n-i]| \cdot k[i]$$
The onset candidate is the peak of this smoothed envelope:
$$n_{env\_peak} = \arg\max_{n} E[n]$$

---

## 3. Quality Gate Metrics

Once a window of audio $y_{clip}[n]$ of length $N$ is extracted around an onset, it must pass rigorous mathematical quality checks. Let the clip be denoted as $y[n]$ for simplicity.

### 3.1 Peak Amplitude
The maximum absolute value in the clip:
$$Peak = \max_{n} |y[n]|$$

### 3.2 Root Mean Square (RMS) Energy
RMS measures the continuous power of the clip.
$$RMS = \sqrt{ \frac{1}{N} \sum_{n=0}^{N-1} (y[n])^2 }$$

### 3.3 Crest Factor
The Crest Factor indicates how extreme the peaks are relative to the overall energy. Gunshots have very high crest factors.
$$Crest = \frac{Peak}{RMS + \epsilon}$$
*(where $\epsilon = 10^{-12}$ prevents division by zero)*

### 3.4 Prominence (Peak to Median Ratio)
This determines if the peak is a loud event standing out from a noisy background.
$$Median_{abs} = \text{median}(|y[0]|, |y[1]|, \dots, |y[N-1]|)$$
$$Prominence = \frac{Peak}{Median_{abs} + \epsilon}$$

### 3.5 Spectral Centroid
The spectral centroid indicates where the "center of mass" of the spectrum is located. High-frequency impulses (like snaps and pops of guns) have high centroids.
Using the magnitude spectrogram $S(k,m)$ and corresponding frequencies $f(k)$:
$$Centroid(m) = \frac{\sum_{k} f(k) \cdot S(k,m)}{\sum_{k} S(k,m)}$$
The overall centroid metric is the mean across all frames:
$$Centroid_{overall} = \frac{1}{M} \sum_{m=0}^{M-1} Centroid(m)$$

### 3.6 Attack Ratio (Impulse Filter for Background Noise)
To ensure the non-gunshot clips do NOT contain hidden impulses, the background trimmer compares the energy in the first half of the clip to the second half.
$$E_{first} = \frac{2}{N} \sum_{n=0}^{(N/2)-1} (y[n])^2$$
$$E_{second} = \frac{2}{N} \sum_{n=N/2}^{N-1} (y[n])^2$$
$$Attack\_Ratio = \frac{E_{first}}{E_{second} + \epsilon}$$

If the Attack Ratio is exceedingly high alongside a high Crest factor, the clip is rejected as an anomaly (potential gunshot).

---

## 4. Normalization and Padding

### 4.1 Peak Normalization
Before writing to a WAV file, the audio is DC-centered and peak-normalized to prevent clipping and ensure consistent volume across the dataset.
$$y_{centered}[n] = y[n] - \mu_y$$
where $\mu_y = \frac{1}{N}\sum_{n} y[n]$

$$y_{norm}[n] = y_{centered}[n] \cdot \left( \frac{0.999}{\max_n |y_{centered}[n]|} \right)$$

### 4.2 Constant Padding
If an extracted clip is shorter than $N$ samples (e.g., event happened at the very end of a file), it is padded with zeros (silence).
For a clip $y_{short}$ of length $L < N$:
$$y_{padded}[n] = \begin{cases} 
y_{short}[n] & \text{for } 0 \le n < L \\
0 & \text{for } L \le n < N 
\end{cases}$$
