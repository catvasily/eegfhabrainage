#!/bin/bash
#SBATCH --account=rpp-doesburg
#SBATCH --time=02:00:00		# init preproc stage 00:35:00 (burnaby) (or 59 for Abbotsford)
				# PREP/ICA step 00:35:00 (burnaby), 02:00:00 abbotsford
#SBATCH --job-name=eegfh_array_job
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G	# 16G for initial preproc step, 4G for PREP/ICA step
#SBATCH --mail-user=amoiseev@sfu.ca
#SBATCH --array=0-99	# should normally be 0-99 

# $SLURM_ARRAY_TASK_ID is 0-based index of a single task
module load python/3.8.10
module load scipy-stack

cd  /home/amoiseev/projects/rpp-doesburg/amoiseev/mycode/python/eegfhabrainage
source mne/bin/activate
cd preprocessing

#python run_filtering_segmentation.py ${SLURM_ARRAY_TASK_ID}
python run_pyprep_ica.py ${SLURM_ARRAY_TASK_ID}
deactivate

