"""
Parameterizing power spectra in periodic and aperiodic components.

Based on code in
 https://github.com/fooof-tools/fooof
and original Nature Neuroscience paper referenced therein
"""
import sys
import os
import os.path as path
import glob
import matplotlib.pyplot as plt
from specparam import SpectralModel, SpectralGroupModel
from run_welch import read_welch_psd

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
pathname = lambda fname: path.join(path.dirname(__file__), fname)

# These are only needed for benchmarking
sys.path.append(path.dirname(path.dirname(__file__))+ "/misc")
from utils import timeit

def main():
    fname = '/data/eegfhabrainage/src-welch/Burnaby/2f8ab0f5-08c4-4677-96bc-6d4b48735da2_src_psd.hdf5'
    ich = 33

    ch_names, freqs, psd = read_welch_psd(fname)   # psd = nchan x nf 

    """
    # Single spectrum
    fm = SpectralModel(peak_width_limits=[1.0, 12.0], max_n_peaks=4, min_peak_height=0.1,
                       peak_threshold=2.0, aperiodic_mode='fixed')

    freq_range = [0.5, 55.]
    fm.report(freqs, psd[ich], freq_range)
    """

    # Multiple spectra
    # Initialize a SpectralGroupModel object, specifying some parameters
    fg = SpectralGroupModel(peak_width_limits=[1.0, 12.0], max_n_peaks=4, min_peak_height=0.1,
                       peak_threshold=2.0, aperiodic_mode='fixed')

    # Fit models across the matrix of power spectra
    # Was:
    # fg.fit(freqs, psd)
    # Use this wrapper to time it:
    do_fit(fg, freqs, psd)

    # Create and save out a report summarizing the results across the group of power spectra
    fg.save_report(file_name='report.png')

    # Save out results for further analysis later
    fg.save(file_name='group_results', save_results=True)
    plt.show()

@timeit
def do_fit(fg, freqs, psd):  # do_fit() is only needed to use @timeit
    fg.fit(freqs, psd, n_jobs = -1)

# ---- Epilogue -----------------------------
if __name__ == '__main__': 
    main()
