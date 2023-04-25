import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from mne import viz
from pyprep.prep_pipeline import PrepPipeline
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from edf_preprocessing import assign_known_channel_types
from edf_preprocessing import _JSON_CONFIG_PATHNAME as first_step_json

JSON_CONFIG_FILE = "pyprep_ica_conf.json"
'''Default name (without a path) for the JSON file with parameter settings for
the PYPREP procedure and ICA artifact removal. This file is expected
to reside in the same folder as the *do_pyprep.py* source file.

'''

_JSON_CONFIG_PATHNAME = os.path.dirname(__file__) + "/" + JSON_CONFIG_FILE
'''Automatically generated fully qualified pathname to the default JSON config file

'''

class Pipeline:
    """The class' aim is preprocessing clean extracted segments of clinical 
    EEG recordings (in EDF format) and make them a suitable input for later analysis 
    and ML applications.

    The class instantiates a preprocessing object which 
    carries a Raw EDF file through the following operations: 
    1. Removes power lines, re-references the channels and identifies bad channels
    as per the PREP pipeline.

    2. Performs ICA on the given EEG segment for both EOG and ECG channels.
    Then the object returns a pre-processed EEG data in raw format.

    **Attributes**

    Attributes:
        conf_dict (dict): a dictionary containing all settings; typically reflects
            contents of the JSON file :data:`JSON_CONFIG_FILE`
        ch_groups (dict): a dictionary containing lists of channels of certain types
        raw (mne.Raw): the MNE Raw EDF object
            
    **Methods**

    """
    # This part of the class description was moved out of a docstring to avoid sphinx warnings
    # about duplicate descriptions. Now only descriptions from the methods themselves will
    # be present in the generated documentation.
    #   prep: applies the PREP pipeline the mark the bad channels
    #   filter_group: filters EOG or ECG channels to be used for ICA artifact removal
    #   ica: performs ICA on the given EEG segment for both EOG and ECG channels
    #   showplot: shows the time domain plot of the given EEG segment
    #   applyPipeline: applies the pipeline (resampling, filtering, applying PREP, performing ICA)
    #       on the given EEG segment
    #   getRaw: returns the preprocessed EEG segment in raw format

    def __init__(self, file_name, *, conf_json = None, conf_dict = None,
                 view_plots = False) -> None:
        """
        Args:
            file_name (str): EDF file pathname
            conf_json (str): pathname of a json file with configuration parameters.
                The default configuration file name is given by :data:`JSON_CONFIG_FILE` constant.
            conf_dict (dict): a dictionary with configurartion parameters.
                If both `conf_json` and `conf_dict` are given, the latter is used.
            view_plots (bool): If `True`, various interactive data plots will be shown.
                Set it to False (default) if processing multiple files. 

        """
        # Load configuration settings
        if conf_json is None:
            conf_json = _JSON_CONFIG_PATHNAME

        if conf_dict is None:
            # Read configuraion from a json file
            with open(conf_json, "r") as fp:
                conf_dict = json.loads(fp.read())

        self.conf_dict = conf_dict

        self.raw = mne.io.read_raw_edf(file_name, infer_types = True, preload=False,
                                       verbose = 'ERROR')	# Get a light Raw object for now

        # Set known channel types. Here we need to use another JSON config file - 
        # one from the initial preprocessing and interval selection step
        self.ch_groups = assign_known_channel_types(self.raw, conf_json = first_step_json)

        if (view_plots):
            self.showplot(title = "Original EDF", show = False)	# Do not pause until processing results are shown too
        
    def filter_group(self, *, what = "eog", view_plots = False) -> None:
        """Filters specified channel group to be used as templates for ICA artifact removal

        Args:
            what (str): currently either "eog" or "ecg" - the channel group name
            view_plots (bool): flag to view plots

        Returns:
            None
        """
        assert what == 'eog' or what == 'ecg'

        self.raw.load_data()	# Ensure data is read in before filtering
				# (even if there will be nothing to filter)

        if len(self.ch_groups[what]) == 0:
            return

        ch_lst = self.ch_groups[what]

        if view_plots:
            self.showplot(title = "Original {}s".format(what.upper()), psd = False, picks = ch_lst,
                          show = False)

        filter_picks = self.conf_dict[what + "_filter_kwargs"]["picks"]
        self.conf_dict[what + "_filter_kwargs"]["picks"] = ch_lst
        self.raw.filter(**self.conf_dict[what + "_filter_kwargs"])
        self.conf_dict[what + "_filter_kwargs"]["picks"] = filter_picks

        if (view_plots):
            self.showplot(title = "Filtered {}s".format(what.upper()), psd = False, picks = ch_lst,
                          show = True)

    def prep(self, *, view_plots = False) -> None:
        """Applies the PREP pipeline to the EEG segment to mark the bad channels

        Args:
            view_plots: boolean value to denote if we want to view plots

        Returns:
            None
        """
        mne.datasets.eegbci.standardize(self.raw)

        # Add a montage to the data
        # Was:
        #	montage_kind = "standard_1005"
        # Use a standard 10 - 20 montage instead:
        montage_kind = self.conf_dict["montage"]

        montage = mne.channels.make_standard_montage(montage_kind)

        # Setting on_missing other than 'raise' here does not
        # change anything because the same call is done again within PrepPipeline()
        # with on_missing set to 'raise'
        self.raw.set_montage(montage, on_missing='raise')

        # Extract some info
        sample_rate = self.raw.info["sfreq"]
        fpwr = self.conf_dict["powerline_frq"]

        prep_params = self.conf_dict["prep"]["prep_params"]
        prep_params["line_freqs"] = np.arange(fpwr, sample_rate / 2, fpwr)

        prep = PrepPipeline(self.raw, prep_params, montage,
                            **self.conf_dict["prep"]["other_kwargs"])
        prep.fit()			# Original Raw is yet unchanged
        self.raw = prep.raw		# Replace original Raw with PREP'd one

        print("Bad channels original: {}".format(prep.noisy_channels_original["bad_all"]))
        print("Bad after reref but before interpolation: {}".format(prep.bad_before_interpolation))
        print("Interpolated channels: {}".format(prep.interpolated_channels))
        print("Bad channels after interpolation: {}".format(prep.still_noisy_channels))
        print("DONE PREP Pipeline to re-reference and remove bad channels")

        if (view_plots):
            self.showplot(title = "After PREP applied", time_series = True, psd = True)


    def ica(self, *, view_plots = False) -> None:
        """Performs ICA on the given EEG segment for both EOG and ECG channels

        Args:
            view_plots: flag to show interactive plots

        Returns:
            None
        """
        init_args = self.conf_dict["ica"]["init"]
        # NOTE: Was: n_components = number of EEG channels
        # However often using n_components = None or 0.99999 results in smaller
        # number of ICs which still fully explain the data. We've set the latter
        # option in JSON to avoid having redundant ICs
        ica = ICA(n_components = init_args["n_components"],
                  random_state = init_args["random_state"],
                  method = init_args["method"],
                  max_iter = init_args["max_iter"],
                  verbose = init_args["verbose"],

                  noise_cov = None,
                  fit_params = None,
                  allow_ref_meg = False
                  )

        fit_args = self.conf_dict["ica"]["fit"]
        ica.fit(self.raw,
                picks=fit_args["picks"],	# picks = None or 'eeg' selects all good EEG channels for fitting
                tstep=fit_args["tstep"],
                verbose=fit_args["verbose"],

                start=None,
                stop=None,
                decim=None,
                reject=None,
                flat=None,
                reject_by_annotation=True
                )

        # Flag to actually apply ICA cleaning to the data
        applyICA = self.conf_dict["ica"]["applyICA"]
        
        if len(self.ch_groups["eog"]) > 0:
            eog_args = self.conf_dict["ica"]["find_bads_eog"]
            eog_indices, eog_scores = ica.find_bads_eog(self.raw,
                                                        measure=eog_args["measure"],
                                                        threshold=eog_args["threshold"],
                                                        verbose=eog_args["verbose"],

                                                        ch_name=None,	# Known EOGs will be used
                                                        start=None,
                                                        stop=None, 
                                                        l_freq=None,	# Skip filtering: already done
                                                        h_freq=None,
                                                        reject_by_annotation=True,
                                                        )
            ica.exclude = eog_indices

            if (view_plots) and eog_indices != []:
                ica.plot_scores(eog_scores)
            
                # plot diagnostics
                ica.plot_properties(self.raw, picks=eog_indices)

                # plot ICs applied to raw data, with EOG matches highlighted
                ica.plot_sources(self.raw, show_scrollbars=True)

                self.showplot(title = "Before removing EOG artifacts", psd = False,
                              show = False)

            if applyICA:
                if len(ica.exclude) > 0:
                    ica.apply(self.raw)     # All defautl args look good forever - so
                                            # keeping the call simple
                    print("Removed EOG Artifacts using ICA")

                    if view_plots:
                        self.showplot(title = "After removing EOG artifacts", psd = False)
                else:
                    print("No clear EOG ICs to remove were found")

        if len(self.ch_groups["ecg"]) > 0:
            ica.exclude = []	# Clear indicies of already excluded EOG ICAs
            # find which ICs match the ECG pattern
            ecg_args = self.conf_dict["ica"]["find_bads_ecg"]
            ecg_indices, ecg_scores = ica.find_bads_ecg(self.raw,
                                                        method=ecg_args["method"],
                                                        measure=ecg_args["measure"],
                                                        threshold=ecg_args["threshold"],
                                                        verbose=ecg_args["verbose"],

                                                        ch_name=None,
                                                        start=None,
                                                        stop=None,
                                                        l_freq=None,	# ECGS are already filtered
                                                        h_freq=None, 
                                                        reject_by_annotation=True)
            ica.exclude = ecg_indices

            if (view_plots) and ecg_indices != []:
                # barplot of ICA component "ECG match" scores
                ica.plot_scores(ecg_scores)

                # plot ICs applied to raw data, with ECG matches highlighted
                ica.plot_sources(self.raw, show_scrollbars=False)

                # ica.apply(self.raw)                
                self.showplot(title = "Before removing ECG artifacts", psd = False,
                              show = False)
                
            if applyICA:
                if len(ica.exclude) > 0:
                    ica.apply(self.raw)
                    print("Removed ECG Artifacts using ICA")

                    if view_plots:
                        self.showplot(title = "After removing ECG artifacts", psd = False)
                else:
                    print("No clear ECG ICs to remove were found")

    def showplot(self, *, title = None, time_series = True, psd = True,
                 picks = None, show = True) -> None:
        """Plot the time courses and / or power spectrum  of the data

        Args:
            title (str): the plot title
            time_series (bool): whether time series should be plotted
            psd (bool): whether power spectrum should be plotted
            picks (str or lst): channels to plot; if `None` (default) -  
                all channels will be plotted
            show (bool): flag to show the plot interactively (execution
                is paused)

        Returns:
            None

        """
        cfg = self.conf_dict

        if picks is None:
            raw = self.raw
            delete_raw = False
        else:
            raw = self.raw.copy().pick(picks)
            delete_raw = True

        if time_series:
            fig = raw.plot(title = title, n_channels=len(self.raw.ch_names),
                          show_scrollbars=True,
                          start = 0,
                          duration = cfg["plot"]["time_window"], 
                          block=False, 
                          scalings = cfg["plot"]["scalings"])

            fig.axes[0].set_title(title)
        
        if psd:
            xscale = 'log' if cfg["plot"]["spect_log_x"] \
                     else 'linear'
            fmin = cfg["plot"]["fmin"]
            fmax= cfg["plot"]["fmax"]

            fig = raw.compute_psd(method='welch',
                                  fmin = fmin,
                                  fmax= fmax,
                                  n_fft = cfg["plot"]["n_fft"]
                                  ).plot(
                                         picks = picks,
                                         amplitude = True,
                                         dB = False,
                                         xscale = xscale,
                                         show = False)
            fig.axes[0].set_title(title)

            if cfg["plot"]["spect_log_x"]:
                first_tick = 1. if np.isclose(fmin, 0) else fmin
            else:
                first_tick = fmin

            fstep = cfg["plot"]["fstep"]
            ticks = [first_tick]
            ticks.extend(np.arange(fstep, fmax, fstep))

            if cfg["plot"]["spect_log_y"]:
                fig.axes[0].set_yscale('log')

            fig.axes[0].xaxis.set_ticks(ticks)

            if delete_raw:
                del raw

            if show:
                plt.show()

    def applyPipeline(self, applyICA = False, view_plots = False) -> None:
        """Applies the pipeline (resampling, filtering, applying PREP, performing ICA)
            on the given EEG segment

        Args:
            applyICA (bool): flag to remove EOG, ECG - related ICs from the signals
            view_plots (bool): flag to show tons of plots; execution will be paused
                each time a plot is shown 

        Returns:
            None
        """
        self.filter_group(what = 'eog', view_plots = False)
        self.filter_group(what = 'ecg', view_plots = False)
        self.prep(view_plots = view_plots)
        self.ica(view_plots = view_plots) 

    def getRaw(self):
        """
        Returns:
            Raw EDF object (preprocessed)
        """
        return self.raw
        
