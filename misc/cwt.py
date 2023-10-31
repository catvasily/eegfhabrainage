# =======================================
# Test program: playing with CWT packages
# =======================================
import sys
import numpy as np
from pactools import Comodulogram
from pactools import simulate_pac
from pactools.dar_model import DAR, extract_driver
import cmath    # Working with complex numbers
import pywt
# https://pywavelets.readthedocs.io/en/latest/
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from welch_logf import welch_logf

print(f"Continuous wavelets: {pywt.wavelist(family=None, kind='continuous')}")

#%% Spectrogram
#freq_input = np.logspace(np.log10(1), np.log10(10), num=10, endpoint=True, base=10.0)
freq_input = np.arange(1,51);
fs = 256    # Sampling rate
band = '2.0'    # bandwith "B"; set B = 2 to have scipy's morlet2
fc = '1.0'      # central frequency "C"; set C = 1 to have scipy's morlet2 with w = 2pi
rec_len = 60   # Record length in seconds
#amp_sin = 1     # Amplitude for sinusoids
#amp_n = 0.1   # amplitude for noise
amp_sin = 0     # Amplitude for sinusoids
amp_n = 1   # amplitude for noise
#wavelet = 'gaus1'
#wavelet = 'gaus8'
#wavelet = 'cgau1'
#wavelet = 'cgau8'
#wavelet = 'morl'
wavelet = 'cmor'
#wavelet = 'fbsp'
#wavelet = 'mexh'
#wavelet = 'shan'

#Complex Morlet wavelet: 
# The expression is
# w(t) = (pi*B)^(-1/2) * exp(-t^2/B) * exp(2*pi*j*C*t)
#
# The scipy's morlet2 wavelet is
# w_sp(t) = pi^(-1/4) * exp(-t^2/2) * exp(j*w*t)
#
# Thus to compare with scipy.cwt one needs to set 'cmor'
# wavelet with B = 2, C = w/2pi, and also MULTIPLY THE
# CWT AMPLITUDE by B^(1/2)pi^(1/4)

nfft = 4096      # nfft per segment for Welch

add_low_freqs = True    # Add two low-frequency signals
normalize = False       # Normalize scalogram on 1/scale
image_interpolation = True # Should be True - otherwise image is distorted on screen

# Color scale limits relative to the data range
#vmin = 0.1
#vmax = 0.9
vmin = 0.
vmax = 1

dt = 1/fs
x = np.arange(0,rec_len,dt)
y01 = amp_sin * np.sin(2*np.pi*x*2)  # 2 Hz
y02 = amp_sin * np.sin(2*np.pi*x*4)  # 4 Hz
y1 = amp_sin * np.sin(2*np.pi*x*40) # 40 Hz
y2 = amp_sin * np.sin(2*np.pi*x*20) # 20 Hz
y = np.concatenate((y2,y1), axis=0, out=None)

if add_low_freqs:
    y += np.concatenate((y01,y02), axis=0, out=None)

rng = np.random.default_rng(99945)
y = y + amp_n*rng.normal(0, 1, y.shape[0])

if wavelet == 'cmor':
    wavelet = f'cmor{band}-{fc}'

'''
# AM: This is original variant, using scale2frequency()
scale_range = fs*pywt.scale2frequency(wavelet, freq_input, precision=8)
coef, freqs=pywt.cwt(y,scale_range,wavelet,1/fs)
'''

# --------- pywt version ---------------------------------------
# AM: it looks that we need frequency2scale() instead
# But results are exactly the same. 
# This is because the functions are identical: they take central frequency of the
# wavelet and divide those on the 2nd argument

wobject = pywt.ContinuousWavelet(wavelet)   # QQQQ
print("\nWavelet central frequency for {} wavelet for scale = 1: {} Hz".format(
    wavelet, pywt.central_frequency(wavelet, precision=8)))

scale_range = pywt.frequency2scale(wavelet, freq_input/fs, precision=8)
print(f"Scales: {scale_range}\n")
coef, freqs=pywt.cwt(y,scale_range,wavelet,1/fs)

# Scale to correspond to scypi's morlet2
coef = coef * np.sqrt(float(band))*np.sqrt(np.sqrt(np.pi)) 

#scale_range = pywt.frequency2scale(wobject, freq_input/fs, precision=8)
#coef, freqs=pywt.cwt(y,scale_range,wobject,1/fs)

if normalize:
    coef = np.einsum('ij,i->ij',coef,1./np.sqrt(scale_range))

matfig = plt.figure(figsize=(10, 4))
pwr = np.absolute(coef)**2
dbpwr = 20*np.log10(np.absolute(coef))
dmin = np.min(dbpwr)
drange = np.max(dbpwr) - dmin

plt.matshow(dbpwr, fignum=matfig.number,
        aspect='auto',
        interpolation = 'antialiased' if image_interpolation else 'none',
        vmin = dmin + vmin*drange, vmax = dmin + vmax*drange,
        cmap = 'inferno', origin = 'upper') 

plt.title(wavelet + f", dB, limits {vmin*100}% - {vmax*100}%, {'' if normalize else 'un'}normalized")
plt.savefig(wavelet + '.png')
# --------- end of pywt version ------------------------------------

""" SSQUEEZEPY
# ----------- ssqueezepy-0.6.4 -------------------------------------
# https://dsp.stackexchange.com/questions/71398/synchrosqueezing-wavelet-transform-explanation/71399#71399
# https://dsp.stackexchange.com/questions/86183/signal-reconstruction-using-scipy-signal-cwt
# https://dsp.stackexchange.com/questions/86068/how-to-validate-a-wavelet-filterbank-cwt/86069#86069
# https://dsp.stackexchange.com/questions/76329/wavelet-center-frequency-explanation-relation-to-cwt-scales/76371#76371

import sys
import ssqueezepy as sqpy
wavelet = sqpy.Wavelet('morlet')
scales = sqpy.experimental.freq_to_scale(freq_input, wavelet, len(y), fs=fs)

Wx, scales = sqpy.cwt(y, wavelet, scales = scales, nv = 32)
ff=sqpy.experimental.scale_to_freq(scales, wavelet, len(y), fs)
pwr = np.absolute(Wx)**2

fig = plt.figure(figsize=(10, 4))
# extent = {left, right, bottom, top} in data units
extent = [x[0],x[-1], ff[0], ff[-1]]
plt.matshow(pwr, fignum=fig.number,
        aspect='auto',
        interpolation = 'antialiased' if image_interpolation else 'none',
        # vmin = dmin + vmin*drange, vmax = dmin + vmax*drange,
        cmap = 'inferno', origin = 'lower', extent = extent) 

#plt.title(wavelet + f", dB, limits {vmin*100}% - {vmax*100}%, {'' if normalize else 'un'}normalized")
plt.show()
sys.exit()  # QQQ
# ----------- end of ssqueezepy version ----------------------------
"""

# --------- scipy version -------------------------------------------
# For scipy morlet wavelet, the wavelet main exponent is
#   E(n) = exp(j*w*n/s),
# where n is the sample number. Therefore to get to Hz:
# t = n/fs, then in real time units
#   E(t) = exp(j*w*(fs/s)*t)
# Thus the central frequency of the wavelet at scale s is:
#   fc = (w/2 pi)*fs/s
#
# NOTE that when w ~ 6 and s = 1 ("mother wavelet") fc ~ fs so the wavelet
# is UNDERSAMPLED. The mininal allowed scale is s = 2.
#
# For a given frequency f, the corresponding scale is chosen so that fc = f, thus
#       s(f) = (w/2pi)*fs/f
# 
# Choice of the number of points for wavelet depends on w. For 
# scale 1, w ~ 6 the wavelet != 0 for 6 consequtive points only and mainly
# shows 1 half-wave (due to undersampling). A good smooth plot is obtained for m = 100,
# s = 16 which corresponds to rel frequency 1/16.
# The scipy.signal.cwt calculates m as follows:
#   m = min(10 * s, len(data)), # and this works for w ~ 6, as shown below.
# One cannot change this expression in scipy.cwt.
# 
# In general, the length "m" for N cycles at scale s is calculated as follows.
# fc = (w/2pi)fs/s;                             # wavelet fc for scale s
# L = N T = N (1/fc) = N (s/fs)*(w/2pi)^-1      # Length in seconds
# 
# m = fs * L = N s*(w/2pi)^-1                   # Number of points given N, w, s
# 
# !!! When w = 2pi, one gets N cycles in exactly N*s points !!!
# 
# But in fact one can use m = 5 * s, because the Morlet wavelet only lasts
# 5 cycles - and is zero beyond those.
# 
# For a particular case of fs = 256, f = 1 Hz, w ~ 6 yields:
#       s = 256
#       m = 10*s = 2560 for 10 cycles

# Assuming complex morlet wavelet:
wavelet_sp = 'scipy-morlet2'
w = 2*np.pi     # Wavelet "omega" parameter
fc0 = fs        # Wavelet central frequency for s = 1: fc = (w/2 pi)*fs
scale_range = fc0/freq_input 

cwt_sp = signal.cwt(y, signal.morlet2, scale_range, w = w)  # nfreq x len(y)

fig_sp = plt.figure(figsize=(10, 4))
pwr_sp = np.absolute(cwt_sp)**2
dbpwr = 20*np.log10(np.absolute(cwt_sp))
dmin = np.min(dbpwr)
drange = np.max(dbpwr) - dmin

plt.matshow(dbpwr, fignum=fig_sp.number,
        aspect='auto',
        interpolation = 'antialiased' if image_interpolation else 'none',
        vmin = dmin + vmin*drange, vmax = dmin + vmax*drange,
        cmap = 'inferno', origin = 'upper') 

plt.title(f"scipy/morlet2, dB, limits {vmin*100}% - {vmax*100}%, {'' if normalize else 'un'}normalized")
plt.savefig(wavelet_sp + '.png')
#plt.show()
# --------- end of scipy version ------------------------------------
# Average over time
spect = np.mean(pwr, axis = 1)
peaks = find_peaks(spect)
spect_sp = np.mean(pwr_sp, axis = 1)
peaks_sp = find_peaks(spect_sp)
"""
plt.figure()
plt.plot(spect)
title = wavelet + f': power spectrum. Peaks: {list(peaks[0])} Hz'
plt.title(title)
plt.savefig(wavelet + '_spectrum.png')
plt.show()
"""
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
axes[0].semilogy(freq_input, spect)
axes[0].set_title(wavelet + f': Peaks: {list(freq_input[peaks[0]])} Hz')
axes[0].set_xticklabels([])

axes[1].semilogy(freq_input, spect_sp)
axes[1].set_title(wavelet_sp + f': Peaks: {list(freq_input[peaks_sp[0]])} Hz')
axes[1].set_xticklabels([])

# Calculate Welch spectrum on a log grid
fl, Pl, nfft_w = welch_logf(y, freq_input[0], freq_input[-1], nf = len(freq_input), fs=fs, nperseg = None,
    scaling='density')
axes[2].semilogy(fl, Pl)
peaks = find_peaks(Pl, height = 1e-2*max(Pl))
axes[2].set_title('Welch on a log scale. Peaks: {} Hz'.format(list(np.round(fl[peaks[0]]))))
plt.savefig(wavelet + '_spectrum.png')
plt.show()

'''Not used
#%%create an artificial signal with PAC.
fs = 200.  # Hz
high_fq = 40.0  # Hz
low_fq = 7.0  # Hz
low_fq_width = 1.0  # Hz
n_points = 1000
noise_level = 0.4
one_signal = simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
                      low_fq_width=low_fq_width, noise_level=noise_level,
                      random_state=0)
'''

