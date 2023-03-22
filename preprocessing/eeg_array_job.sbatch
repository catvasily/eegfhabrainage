#!/bin/bash
#SBATCH --account=rpp-doesburg
#SBATCH --time=00:25:00
#SBATCH --job-name=eegfh_array_job
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=amoiseev@sfu.ca
#SBATCH --array=0-99 

# $SLURM_ARRAY_TASK_ID is 0-based index of a single task
module load python/3.8.10
module load scipy-stack

cd  /home/amoiseev/projects/rpp-doesburg/amoiseev/mycode/python/eegfhabrainage
source mne/bin/activate
cd preprocessing

python test_preproc.py ${SLURM_ARRAY_TASK_ID}
deactivate

