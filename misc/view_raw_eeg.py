# -----------------------------------------------
# A helper to view a plot of the raw data
#
# Can be used standalone, or as a function call
# -----------------------------------------------
import tkinter as tk
from tkinter import filedialog

import mne
from mne.io import read_raw_edf, read_raw_fif

BACKEND = 'matplotlib'		# Backend to use: 'matplotlib' or 'qt'
DURATION = 30			# Time window width

# --- Inputs: ----
# file_name = "800c7738-239b-46db-9612-328411369a9d.edf"
file_name = None
path = "/data/eegfhabrainage/hv_segments/Abbotsford/"
#path = "/data/eegfhabrainage/Abbotsford/"

if file_name is None:
	dsname = None
else:
	dsname = path + '/' + file_name 

picks = None
highpass=None
lowpass=None
raw = None
exclude = None
include = None
# ----------------

# -------------------------------------------------------------------------------------------
def view_raw_eeg(*, dsname = None, raw = None, picks = None, highpass = None, lowpass = None,
			exclude = None, include = None):
# -------------------------------------------------------------------------------------------
	mne.viz.set_browser_backend(BACKEND)

	if exclude is None:	# Default excludes based on FH project
		exclude = ["AUX1", "AUX2", "AUX3", "AUX4", "AUX5", "AUX6", "AUX7", "AUX8", "DC1", "DC2",
			     "DC3",  "DC4",  "DIF1", "DIF2", "DIF3", "DIF4",
			     "", "L SPH", "R SPH",
			     "aux1", "aux2", "aux3", "aux4", "aux5", "aux6", "aux7", "aux8", "dc1", "dc2",
			     "dc3",  "dc4",  "dif1", "dif2", "dif3", "dif4",
			     "l sph", "r sph",
                             "Patient Event", "Photic", "Trigger Event", "x1", "x2",
			     "phoic", "Phoic", "photic", 
			     "PATIENT EVENT", "PHOTIC", "TRIGGER EVENT", "X1", "X2",
			     "PHOIC", "PHOTIC"]

	# 'raw' takes precedence over 'dsname'; one of them should be present
	if raw is None:
		if dsname is None:
			raise ValueError("Either the 'dsname' or the 'raw' argument should be supplied")
		else:
			ext = dsname[-3:].upper()

			if ext == 'EDF':
				raw = read_raw_edf(dsname, eog=None, misc=None, stim_channel='auto', 
					exclude=exclude,
					infer_types=False,
					include=None,		# Set exclude to () if include is used
					preload=True,
					units=None,		# Those stored in file will be used
					encoding='utf8',	# Encoding of annotations
					verbose=None)
			elif ext == 'FIF':
				raw = read_raw_fif(dsname, preload=True,
					verbose=None)
			else:
				raise ValueError("Only EDF and FIF files are currently supported.")


	mne.viz.plot_raw(raw.pick(picks = picks),
		events=None, duration=DURATION, start=0.0, n_channels=20,
		bgcolor='w', color=None, bad_color='lightgray', event_color='cyan', scalings=None,
		remove_dc=False,	# default is True
		order=None,		# Order in which channels are plotted. None = all
		show_options=True, 	# default False. Show dialog about projections
		title=None,
		show=True,
		block=True,		# Stop execution until the plot is dismissed
		highpass=highpass,		# Default None
		lowpass=lowpass,
		filtorder=4,		# Default 4; 0 uses FIR filter
		clipping=None,		# Default = 1.5
		show_first_samp=True,	# default False. If True, the actual number
					# of a sample which is designated as first is shown
		proj=True,
		group_by='type',
		butterfly=False,	# Start in butterfly mode
		decim='auto', noise_cov=None, event_id=None,
		show_scrollbars=True, show_scalebars=True,
		time_format='float', precompute=None, use_opengl=None, theme=None,
		overview_mode='channels',	# default None
		verbose=None)

if __name__ == '__main__':
	root = tk.Tk()
	root.withdraw()

	dsname = filedialog.askopenfilename()
	view_raw_eeg(dsname = dsname, raw = raw, picks = picks, highpass = highpass, lowpass = lowpass,
			exclude = exclude, include = include)

