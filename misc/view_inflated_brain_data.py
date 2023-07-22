import os.path as path
import sys
import numpy as np
import mne

sys.path.append(path.dirname(path.dirname(__file__))+ "/preprocessing")
from do_src_reconstr import parse_vertex_list, encode_vertex_list

def plot_vertices(brain, hemi, vertno,
        scale_factor = 0.2,
        color = 'white',
        alpha = 1.,
        resolution = 50
    ):
    """Display points specified by vertex numbers on the brain surface

    Args:
        brain (Brain): an instance of MNE Brain class
        hemi (str): the hemisphere from which to read the parcellation,
            can be 'lh', 'rh', or 'both'
        vertno (ndarray): 1D signed integer array for COM vertex numbers, with
            NEGATIVE numbers referring to the LEFT hemisphere, and non-negative
            (including 0) - to the right hemisphere. Negative vertex with number
            _LEFT_HEMI_ZERO is interpreted as vertex 0 of the left hemisphere.
        scale_factor (float): size of plotted dots relative to 1 cm
        color (color): color of dots in any matplotlib form: string, RGB, hex, etc
        alpha (float): opacity in [0, 1] interval
        resolution (int): the resolution of the dot spheres

    Returns:
        None

    """
    do_hemi = (True, True) if hemi == 'both' else (hemi == 'lh', hemi == 'rh')
    map_surface = "white"

    lhv, rhv, lh_idx, rh_idx = parse_vertex_list(vertno)
    vertices = [lhv, rhv]

    for ih, hms in enumerate (('lh', 'rh')):
        if not do_hemi[ih]:
            continue

        brain.add_foci(
            coords = vertices[ih], coords_as_verts=True,
            map_surface=map_surface,
            scale_factor=scale_factor,
            color=color, alpha=alpha,
            name=None,
            hemi=hms,
            resolution=resolution
        )

def plot_rr(brain, hemi, rr,
        map_to_cortex = False,
        scale_factor = 0.2,
        color = 'white',
        alpha = 1.,
        resolution = 50
    ):
    """Display points specified by 3D spatial positions on the brain surface

    Args:
        brain (Brain): an instance of MNE Brain class
        hemi (str): the hemisphere from which to read the parcellation,
            can be 'lh', 'rh', or 'both'
        rr (ndarray): `npoints x 3` array of MRI coordinates, in meters. It is assumed that
            points with negative `x` are referring to the LEFT hemisphere, and non-negative
            (including 0) - to the right hemisphere. Generally this is not exactly so
            but should be correct for the `fsaverage` template.
        map_to_cortex (Bool): flag whether foci will be mapped to the closest vertex
            on the cortical surface, or left as is. Ignored when vertno is provided.
            Note that if `map_to_cortex == False` rr's will never land onto the
            inflated surface even if they do belong to the original (not inflated) cortical
            surface.
        scale_factor (float): size of plotted dots relative to 1 cm
        color (color): color of dots in any matplotlib form: string, RGB, hex, etc
        alpha (float): opacity in [0, 1] interval
        resolution (int): the resolution of the dot spheres

    Returns:
        None

    """
    do_hemi = (True, True) if hemi == 'both' else (hemi == 'lh', hemi == 'rh')
    map_surface = "white" if map_to_cortex else None

    lh_idx = rr[:,0]<0
    lhv = rr[lh_idx]
    rh_idx = np.logical_not(lh_idx)
    rhv = rr[rh_idx]

    vertices = [lhv, rhv]

    for ih, hms in enumerate (('lh', 'rh')):
        if not do_hemi[ih]:
            continue

        # Mind that default units choice for Brain class is mm
        brain.add_foci(
            coords = 1e3*vertices[ih], coords_as_verts=False,
            map_surface=map_surface,
            scale_factor=scale_factor,
            color=color, alpha=alpha,
            name=None,
            hemi=hms,
            resolution=resolution
        )

def expand_data_to_rois(data, labels):
    """Expand ROI single data values to all ROI vertices.

    For example, when one has one value per ROI in the atlas, these values
    will be properly extended to all vertices of hemisphere surfaces.

    Args:
        data (ndarray): 1D array, `shape = (nlabels,)`; data values
            assigned to ROIs (labels)
            vertex values
        labels (list): a list of `nlabels` ROI Label objects

    Returns:
        vdata (ndarray): 1D array; data values for all vertices in all labels
        vertices (ndarray): 1D array; vertex numbers for each data value
    """
    if len(data) != len(labels):
        raise ValueError("Lenghts of data and labels list must be equal")

    vertices = list()
    vdata = list()
    for i,l in enumerate(labels):
        vtc = encode_vertex_list(l.vertices, 'lh' in l.name)
        vertices.extend(vtc)
        vdata.extend([data[i]]*len(vtc))

    return np.array(vdata), np.array(vertices)

def view_inflated_brain_data(
        brain = None,
        atlas = None,
        show_atlas = False,
        hemi = 'both',
        title = None,
        data = None,
        cbar_lims = None,
        colormap = 'auto',
        alpha_data = 0.25,
        smoothing_steps = None,
        expand_values_to_roi = False,
        rois_to_mark = None,
        vertno = None,
        show_vertices = False,
        rr = None,
        map_to_cortex = False,
        subjects_dir = None,
        scale_factor = 0.2,
        color_dots = 'white',
        alpha_cortex = 1.,
        resolution = 50,
        show = True,
        block = False,
        inflated = True,
        kwargs_brain = None,
        kwargs_data = None,
        verbose = None
    ):
    """A utility for displaying data on the brain surface

    Args:
        brain (Brain or None): an instance of the Brain class. If supplied,
            it will be used for plotting, otherwise a new one will be created.
        atlas (str or None): name of the parcellation to diplay
        show_atlas (bool): if `True` parcellation will be displayed
        hemi (str): the hemisphere from which to read the parcellation,
            can be 'lh', 'rh', or 'both'.
        title (str or None): a title for the plot.
        data (ndarray or None): 1D array of `nvertno` floating point values, where 
            `nvertno = len(vertno)`. If data is not `None`, `vertno` should also
            be given. `data` and `vertno` should have the same length.
        cbar_lims ((float,float) or None): (fmin, fmax) color bar limits for the data.
            If `None`, those will be set automatically.
        colormap (str, list of color, or array): as is. Typically a string defining
            the palette name, or 'auto'. See
            *https://mne.tools/stable/generated/mne.viz.Brain.html#mne.viz.Brain.add_data*
            for details and other options.
        alpha_data (float): opacity of the data overlay on the cortex surface
        smoothing_steps (int or 'Nearest' or None): smoothing when plotting data. `int`
            value explicitly sets the number of vertices for smoothing. For other
            settings see
            *https://mne.tools/stable/generated/mne.viz.Brain.html#mne.viz.Brain.add_data*
        expand_values_to_roi (bool): If `True` then all ROI vertices will be assigned
            the same data value as that of the reference vertex. Default is `False`.
        rois_to_mark (list or None): if supplied, a list of names of ROIs (labels)
            or a list of Label objects. In the first case `atlas` cannot be `None`,
            and the name should belong to this atlas. In the 2nd case a ROI from
            any surface atlas can be displayed; mind not using `Label` objects
            restricted to a (coarse) source space - those won't show up correctly.
            The specified ROIs will be shown with thicker borders.
        vertno (ndarray or None): 1D signed integer array for COM vertex numbers, with
            NEGATIVE numbers referring to the LEFT hemisphere, and non-negative
            (including 0) - to the right hemisphere. Negative vertex with number
            _LEFT_HEMI_ZERO is interpreted as vertex 0 of the left hemisphere.
        show_vertices (bool): If `True`, the vertices pointed to by vertno will be
            plotted. Default `False`.
        rr (ndarray or None): `nr x 3`; a list of locations in MRI coordinates to plot. Those
            will be projected to the nearest cortex location if `map_to_cortex` is `True`.
            Importantly, **`rr`'s are NOT "inflated" when plotted**. Specifically, **if
            the inflated surface is used (Default), and `map_to_cortex` is `False`,
            then points that actually belong to the surface will be displayed off
            the surface**. In view of this an exception is thrown when trying to
            plot rr's with `inflated = True` and `map_to_cortex = False`. Use a
            non-inflated surface to properly display rr's relative to the brain.
        map_to_cortex (bool): flag whether rr's will be mapped to the closest vertex
            on the cortical surface, of left as is. 
        subjects_dir (str): a pathname to the folder containing the
            FreeSurfer subjects reconstructions.
        scale_factor (float): size of plotted dots relative to 1 cm
        color_dots (color): color of plotted spheres (dots) for vertices (if shown)
            or rr's, in any matplotlib form: string, RGB, hex, etc
        alpha_cortex (float): opacity of the cortex surface and the dots (spheres) if
            vertices or rr's are plotted; should be in [0, 1] interval
        resolution (int): the resolution of the dot spheres
        show (bool): Display the window as soon as it is ready. Defaults to `True`.
        block (bool): If `True`, start the Qt application event loop. Default to `False`.
        inflated (bool): If `True`, (default) show inflated pial surface.
        kwargs_brain (dict or None): if needed, more arguments to the Brain class constructor
            in addition to those listed above; see
            *https://mne.tools/stable/generated/mne.viz.Brain.html*
            for details.
        kwargs_data (dict) or None: if needed, more arguments to the Brain.add_data() method
            in addition to those listed above; see
            *https://mne.tools/stable/generated/mne.viz.Brain.html#mne.viz.Brain.add_data* 
            for details.
        verbose (str or None): verbosity level; one of ‘DEBUG’, ‘INFO’,
            ‘WARNING’, ‘ERROR’, ‘CRITICAL’ or None

    Returns:
        brain (Brain): the new or old instance of the Brain class

    """
    BRAIN_FIG_DEFAULTS = { 
        'cortex': "low_contrast",
        'size': (800, 600),
        'background': "white",
        'units': 'mm'
    }

    def _set_kwargs_brain():
        nonlocal kwargs_brain

        if kwargs_brain is None:
            kwargs_brain = BRAIN_FIG_DEFAULTS

        for key in BRAIN_FIG_DEFAULTS:
            if key not in kwargs_brain:
                kwargs_brain[key] = BRAIN_FIG_DEFAULTS[key]

    def _load_labels():
        labels = mne.read_labels_from_annot(
            "fsaverage",
            parc = atlas,
            hemi = hemi,
            surf_name='white', 
            subjects_dir=subjects_dir,
            sort=True,                       # Sort labels in alphabetical order
            verbose=verbose
        )
        return np.array(labels)

    fmin, fmax = cbar_lims if cbar_lims is not None else (None, None)

    surface = "white" if not inflated else "inflated"

    if (atlas is None) and show_atlas:
        raise ValueError('show_atlas is True but no atlas specified')

    if data is not None:
        if vertno is None:
            raise ValueError("vertno should be specified when the data argument is not None")

        if len(data) != len(vertno):
            raise ValueError("len(data) should be equal to len(vertno)")

    labels = None

    if rois_to_mark is not None:
        all_str = all([isinstance(r, str) for r in rois_to_mark])

        if all_str:
            if atlas is None:
                raise ValueError("atlas is not specified but names of ROIs to show are given")

            labels = _load_labels()
            labels_to_mark = [l for l in labels if l.name in rois_to_mark]

        else:
            all_labels = all([isinstance(r, mne.Label) for r in rois_to_mark])

            if not all_labels:
                raise ValueError("rois_to_mark must be either a list of strings or a list of Label objects")

            labels_to_mark = rois_to_mark
    else:
        labels_to_mark = None

    if brain is None:
        Brain = mne.viz.get_brain_class()    # Get a proper Brain class depending on backend used
        _set_kwargs_brain()

        brain = Brain(
            "fsaverage",
            hemi = hemi,
            surf = surface,
            title = title,
            alpha = alpha_cortex,
            subjects_dir=subjects_dir,
            show = show,
            block = block,
            **kwargs_brain
        )

    # Plot the data overlay
    if data is not None:
        lhv, rhv, lh_idx, rh_idx = parse_vertex_list(vertno)
        # !!! AMBUSH !!! in brain.add_data():
        # when
        #	len(vertices) = total # of vertices in the hemisphere,
        # the vertices parameter is IGNORED, and the data->vertex mapping is assumed to be in
        # the order of the surface vertices - and what exactly is this order, anybody?
        # To avoid this pitfall and to force the add_data() to always respect vertices argument,
        # make sure that this equality never happens:
        clip_data = lambda hemi, data, vtc: \
            (data, vtc) if len(brain.geo[hemi].x) != len(vtc) else (data[:-1], vtc[:-1])

        kwa = {'fmin': fmin, 'fmax': fmax, 'alpha': alpha_data, 'smoothing_steps': smoothing_steps,
                'colormap': colormap, 'verbose': verbose}

        if kwargs_data is not None:
            kwa.update(kwargs_data)

        # Add the data to each hemisphere separately
        hemi_args = {'lh': (lhv, lh_idx), 'rh': (rhv, rh_idx)}
        for hh in hemi_args:
            if not (hh in brain.geo):
                continue

            vtc, h_idx = hemi_args[hh]
            arr, vtc = clip_data(hh, data[h_idx], vtc)
            brain.add_data(arr, vertices = vtc, hemi = hh, **kwa)

    if (atlas is not None) and show_atlas:
        brain.add_annotation(atlas)

    if (vertno is not None) and show_vertices:
        plot_vertices(brain, hemi, vertno,
            scale_factor = scale_factor,
            color = color_dots,
            alpha = alpha_cortex,
            resolution = resolution
        )

    if rr is not None:
        if (not map_to_cortex) and inflated:
            raise ValueError("Plotting rr's with an inflated cortex surface makes sense only if map_to_cortex = True")

        plot_rr(brain, hemi, rr,
            map_to_cortex,
            scale_factor = scale_factor,
            color = color_dots,
            alpha = alpha_cortex,
            resolution = resolution
        )
    
    # Mark specific ROIs if requested
    if labels_to_mark is not None:
        borders = 2 if not show_atlas else 4

        for l in labels_to_mark:
            brain.add_label(l, borders=borders, color = 'r',
                reset_camera = False)

    return brain

# Unit test    
if __name__ == '__main__':
    from mne.transforms import apply_trans, invert_transform
    from do_src_reconstr import get_label_coms, get_voxel_coords 

    user_home = path.expanduser("~")
    user = path.basename(user_home) # Yields just <username>

    # --- Inputs ------
    atlas = "aparc.a2009s"
    show_atlas = False
    hemi = 'both'
    #hemi = 'lh'
    fwd_file = '/user/' + user + '/data/eegfhabrainage/src-reconstr/Burnaby/2f8ab0f5-08c4-4677-96bc-6d4b48735da2/2f8ab0f5-08c4-4677-96bc-6d4b48735da2-ico-3-fwd.fif'
    map_to_cortex = False
    alpha_cortex = 1 
    alpha_data = 0.25
    smoothing_steps = 3
    inflated = False
    expand_values_to_roi = True
    show_vertices = False
    plot_rr = False	# Flag to plot ROI COMs via their 3D coords (as opposed to vertex #)
                        # This won't work for inflated brain surface 
    verbose = 'INFO'
    kwargs_data = {'fmid': 0, 'transparent': True}
    # -- end of inputs --

    fs_dir = user_home + '/mne_data/MNE-fsaverage-data'

    # Get COMs for ROIs defined on a coarse surface corresponding
    # to a source space
    fwd = mne.read_forward_solution(fwd_file)
    sspaces = fwd['src']                 # These source spaces are in HEAD coords

    mri_labels = mne.read_labels_from_annot( # These labels are on a dense surface
        "fsaverage",
        parc = atlas,
        hemi = hemi,
        surf_name='white', 
        subjects_dir=fs_dir,
        sort=True,                       # Sort labels in alphabetical order
        verbose=verbose
    )

    visual_rois = [label for label in mri_labels \
        if 'occip' in label.name]
    roi_names = [l.name for l in visual_rois]

    # Use this snippet to plot 'coarse' ROIs which contain
    # only source space vertices
    #labels = [l.restrict(sspaces) for l in mri_labels] # These are 'coarse' labels
    #labels = [l for l in labels if len(l.vertices)]    # Drop labels with no vertices
    #del mri_labels

    # Alternatively, use this line for full resolution (FreeSurfer) ROIs
    labels = mri_labels

    label_coms = get_label_coms(labels, fs_dir)
    label_com_rr = get_voxel_coords(sspaces, label_coms)    # rr's will be in head coords

    data = np.zeros((len(label_coms)))
    roi_labels = list()
    roi_coms = list()
    for i,l in enumerate(labels):
        if l.name in roi_names:
            data[i] = 1.
            roi_labels.append(l)
            roi_coms.append(label_coms[i])

    if expand_values_to_roi:
        data, vertno = expand_data_to_rois(data, labels)
    else:
        vertno = label_coms

    roi_coms = np.array(roi_coms)
    roi_rr = get_voxel_coords(sspaces, roi_coms)

    # Convert rr's to MRI coords
    trans = invert_transform(fwd['mri_head_t'])
    label_com_rr = apply_trans(trans, label_com_rr, move=True)
    roi_rr = apply_trans(trans, roi_rr, move=True)

    for i in range(len(roi_coms)):
        print('roi: {}, rr = [{:0f},{:0f},{:0f}]'.format(roi_names[i], \
            *(1e3*roi_rr[i])))

    brain = view_inflated_brain_data(
        atlas = atlas,
        show_atlas = show_atlas,
        hemi = hemi,
        data = data,
        alpha_data = alpha_data,
        smoothing_steps = smoothing_steps,
        rois_to_mark = visual_rois,
        #rois_to_mark = roi_names,
        vertno = vertno,
        #vertno = roi_coms,
        show_vertices = show_vertices,
        map_to_cortex = map_to_cortex,
        subjects_dir = fs_dir,
        color_dots = 'white',
        block = False,
        alpha_cortex = alpha_cortex,
        inflated = inflated,
        verbose = verbose)

    # When map_to_cortex = False, rr's will be OFF the inflated
    # surface but will match uninflated surface
    if plot_rr:
        view_inflated_brain_data(brain, atlas = atlas, hemi = hemi,
            rr = label_com_rr,
            map_to_cortex = map_to_cortex,
            subjects_dir = fs_dir,
            color_dots = 'red',
            block = True,
            inflated = inflated,
            verbose = verbose)

    input("Press ENTER to continue...")

    # Save a screenshot
    brain.save_image("screenshot.png")

