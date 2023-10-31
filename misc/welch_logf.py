import numpy as np
from numpy import log10
import scipy
from scipy.signal import welch
from warnings import warn
from utils import next_power_of_2, closest_elements

def welch_logf(x, fmin, fmax, nf = 100, fs=1.0, nperseg = None,
    scaling='density', **kwargs):
    """Calculate power spectrum by Welch method for logarithmically
    spaced frequencies.

    Procedure. Given desired f-range `[fmin, fmax]` and number of points
    `nf` (including the ends of the interval), the step in `log(f)` is

    `dlf = (log10(fmax) - log10(fmin))/(nf -1)`

    The necessary frequency step in the linear FFT is determined by the
    lowest end of the spectrum as

    `df = 10^(log10(fmin) + dlf) - fmin`

    To obtain this frequency resolution, the lenght of the FFT for each
    segment in Welch method should be

    `nperseg = fs/df`

    where `fs` is the sampling frequency. After the PSD for linear frequency
    set is found, the PSD value for each non-linear frequency is calculated in
    two steps:
    * find the closest linear frequency
    * assign this a value equal to the average PSD over all linear freqs that
      fall into the interval between this non-linear freq and preceding non-
      linear frequency

    Args:
        x (array-like): Time series of measurement values. If x is an array,
            the spectrum is calculated along the last dimension (axis = -1)
        fmin (float): Min frequency boundary (Hz); > 0
        fmax (float): Max frequency boundary (Hz); > 0
        nf (int > 1): number of frequency points equally spaced in logarithmic
            scale
        fs (float): sampling frequency, Hz
        nperseg (int or None): If `None`, first `nperseg0` will be calculated as
            described above then increased to the closest power of 2. If
            specified and turns out to be less than `nperseg0` - a warning
            will be issued.
        scaling (‘density’ or ‘spectrum’): Selects between computing the power
            spectral density (‘density’) where `Pxx` has units of `V**2/Hz` and
            computing the power spectrum (‘spectrum’) where `Pxx` has units of
            `V**2`, if `x` is measured in `V` and `fs` is measured in Hz.
        kwargs (dict): Additional arguments (if any) to pass to scipy's welch()
            function. One of `window, nooverlap, nfft, detrend, return_onesided,`
            'average'. See
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
            for details.
    
    Returns:
        f (ndarray): Array of sample frequencies, Hz (log-spaced)
        Pxx (ndarray): Power spectral density or power spectrum of `x`
        nfft (int) : size of the linear FFT that was subsampled to get log freqs

    """
    if (fmin >= fs/2) or (fmax > fs/2):
        raise ValueError('fmin must be less, and fmax must be less or equal to fs/2')

    if (fmin <= 0) or (fmax <=0) or np.isclose(fmin, fmax):
        raise ValueError('Both fmin and fmax must be positive and not equal')

    if nf <= 1:
        raise ValueError('nf should be larger than 1')

    if fs <= 0:
        raise ValueError('fs should be positive')

    if nperseg is not None:
        if nperseg < 2:
            raise ValueError('nperseg cannot be less than 2')

    if fmin > fmax:
        tmp = fmax
        fmax = fmin
        fmin = tmp

    lfmin = log10(fmin)
    lfmax = log10(fmax)
    # Get the step in log scale, and equally spaced log10s 
    lf, dlf = np.linspace(lfmin, lfmax, nf, endpoint = True, retstep = True)

    # Target log-spaced frequencies:
    fl = 10**lf

    # Averaging intervals boundaries
    fl0 = 10**(lf - dlf/2); fl0[0] = fl[0]    # Lower ends
    fl1 = 10**(lf + dlf/2); fl1[-1] = fl[-1]  # Upper ends

    # df is the frequency precision we need
    df = 10**(lfmin + dlf) - fmin

    # This is the min FFT length to achieve this precision
    nperseg0 = int(np.ceil(fs/df))

    if nperseg is None:
        nperseg = next_power_of_2(nperseg0)
    else:
        if nperseg < nperseg0:
            warn('Specified nperseg value {} is less than the estimated minimum \
setting for this parameter {}'.format(nperseg, nperseg0))

    # Calculate the spectrum on a linear grid
    f, P = welch(x, fs=fs, nperseg=nperseg, scaling=scaling, axis = -1, **kwargs)

    # Pick values from f which are closest to the ends of averaging intervals
    # ifl = closest_elements(f, fl) QQQQ not needed
    ifl0 = closest_elements(f, fl0)
    ifl1 = closest_elements(f, fl1)

    # Calculate spectrum values as means over corresponding intervals around fl[]
    Pl = list()
    for i in range(len(ifl0)):
        j0 = ifl0[i]
        j1 = ifl1[i] + 1    # +1 in order to include j1 in averaging
        Pmean = np.mean(P[...,j0 : j1], axis = -1)
        Pl.append(Pmean)

    return fl, np.array(Pl), nperseg

# Unit test    
# Use scipy's example for welch()
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    rng = np.random.default_rng()

    # ---- Inputs -----
    fs = 256.
    N = int(1*8192)      # number of x points
    nperseg = 256        # nfft for linear frequencies
    amp = np.sqrt(2)     # sine wave amplitude
    freq1 = 10.0          # sine wave frequency
    freq2 = 60.0          # sine wave frequency

    fmin = 1.            # fmin for log-spaced f
    fmax = 128.          # fmax for log-spaced f
    nf = 128             # number of points for log f

    # Max FFT size to use for log f, or None. If None, it
    # will be selected automatically but may be very large
    # spacing between log lines close to fmin is small
    #nfft_max = 512
    nfft_max = None
    noise_power = 0.001 * fs / 2
    ylim = [1e-4, 2]
    marker = '.'         # Marker to print log f points
    # -- end of inputs ---

    time = np.arange(N) / fs
    x = amp*np.sin(2*np.pi*freq1*time)
    x += amp*np.sin(2*np.pi*freq2*time)
    x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    
    # Plot standard spectrum density first
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    f, Pxx_den = welch(x, fs, nperseg=nperseg)
    axes[0].semilogy(f, Pxx_den)
    axes[0].set_ylim(ylim)
    axes[0].set_xlabel('frequency [Hz]')
    axes[0].set_ylabel('PSD [V**2/Hz]')
    axes[0].set_title('Linear f, nfft = {}'.format(nperseg))
    
    # Plot for log-spaced grid
    fl, Pl, nfft_w = welch_logf(x, fmin, fmax, nf = nf, fs=fs, nperseg = nfft_max,
        scaling='density')
    f, Pxx_den = welch(x, fs, nperseg=nfft_w)	# Linear grid PSD that was subsampled
    axes[1].semilogy(f, Pxx_den, 'g')
    axes[1].set_xlabel('frequency [Hz]')
    axes[1].set_ylabel('PSD [V**2/Hz]')
    axes[1].set_title('Linear f spectrum to sample for log f, nfft = {}'.format(nfft_w))

    axes[2].semilogy(fl, Pl, 'r')
    # axes[2].semilogy(fl, Pl, marker)
    axes[2].set_ylim(ylim)
    axes[2].set_xlabel('frequency [Hz]')
    axes[2].set_ylabel('PSD [V**2/Hz]')
    axes[2].set_title('Log f, nf = {}'.format(nf))

    plt.tight_layout()
    plt.show()

