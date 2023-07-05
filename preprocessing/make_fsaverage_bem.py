import os.path as path
from warnings import warn
import mne

def make_fsaverage_bem(subjects_dir, ico_bem = 4, ico_src = 3, verbose = None):
    """Create BEM model, BEM solution and surface
    source space for the template 'fsaverage' subject.

    fsaverage's BEMs and source space are normally supplied with the 
    fsaverage data for ico = 4 (5120) which produces rather dense source
    space. This function can recreate these but the main purpose is to
    generate lower resolution BEMs and source spaces (ico = 3) because for low
    density EEGs the solutions supplied by have too high resolution.
    Standard condactivities set (0.3, 00., 0.3) is used for the 3-layers EEG
    forward modeling.

    The calculated BEM / source spaces are saved to the '.../fsaverage/bem' folder,
    unless those already exist. In this case a warning is issued and calculation is
    skipped.

    Args:
        subjects_dir (str): full path to the subjects' folder containing fsaverage
            subject
        ico_bem (int): ico setting for BEM model: 3, 4, or 5
        ico_src (int): ico setting for the source space: 3, 4, or 5
        verbose (str or None): verbose mode to use

    Returns:
        None
    """
    ico_dict = {3: 1280, 4: 5120, 5: 20484}
    ico_str = '-{}-{}-{}-'.format(ico_dict[ico_bem],ico_dict[ico_bem],ico_dict[ico_bem])

    bem_sol_pathname = subjects_dir + "/fsaverage/bem/" + "fsaverage{}bem-sol.fif".format(ico_str)
    src_space_pathname = subjects_dir + "/fsaverage/bem/" + "fsaverage-ico-{}-src.fif".format(ico_src)

    # BEM
    if path.isfile(bem_sol_pathname):
        warn("Skipping BEM solution computation for ico = {}: file already exists".format(ico_bem))
    else:
        model =  mne.make_bem_model('fsaverage', ico=ico_bem, conductivity=(0.3, 0.006, 0.3),
            subjects_dir=subjects_dir, verbose=verbose)        
        bem = mne.make_bem_solution(model, verbose = verbose)
        mne.write_bem_solution(bem_sol_pathname, bem, overwrite=False, verbose=verbose)

    # Source space
    if path.isfile(src_space_pathname):
        warn("Skipping source space construction for ico = {}: file already exists".format(ico_src))
    else:
        src = mne.setup_source_space(
            'fsaverage',
            spacing='ico{}'.format(ico_src),
            surface='white',
            subjects_dir=subjects_dir, add_dist=True,
            n_jobs=-1, 
            verbose=verbose)

        mne.write_source_spaces(src_space_pathname, src, 
            overwrite=True, verbose=verbose)

if __name__ == '__main__':
    # ------- Input ----------
    user_home = path.expanduser("~")
    user = path.basename(user_home) # Yields just <username>

    # subjects_dir = user_home + '/mne_data/MNE-fsaverage-data'    # local
    subjects_dir = user_home + '/projects/rpp-doesburg/' + user + '/data/mne_data/MNE-fsaverage-data'
    ico_bem = 4
    ico_src = 3    
    # ------------------------

    make_fsaverage_bem(subjects_dir, ico_bem = ico_bem, ico_src = ico_src, verbose = None)

