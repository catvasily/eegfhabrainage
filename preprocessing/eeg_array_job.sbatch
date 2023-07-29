#!/bin/bash
#SBATCH --account=rpp-doesburg
#SBATCH --time=00:40:00		# init preproc/HV stage 35:00 (burnaby) (or 59 for Abbotsford)
				# PREP/ICA step 00:35:00 (burnaby), 02:00:00 abbotsford.
#SBATCH --job-name=eegfh_array_job	# src reconstr 20 min (burnaby) 40 min others
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G	# 16G for initial preproc step/HV step, 4G for PREP/ICA,
#SBATCH --mail-user=amoiseev@sfu.ca	# 2G for src reconstr
#SBATCH --array=0-99	# should normally be 0-99 

# $SLURM_ARRAY_TASK_ID is 0-based index of a single task
module load python/3.8.10
module load scipy-stack

cd  /home/amoiseev/projects/rpp-doesburg/amoiseev/mycode/python/eegfhabrainage
source mne/bin/activate
cd preprocessing

#python run_filtering_segmentation.py ${SLURM_ARRAY_TASK_ID}
#python run_pyprep_ica.py ${SLURM_ARRAY_TASK_ID}
#python extract_hv_intervals.py ${SLURM_ARRAY_TASK_ID}
python run_src_reconstr.py ${SLURM_ARRAY_TASK_ID}
deactivate

