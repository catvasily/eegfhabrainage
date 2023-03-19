#!/bin/bash

#SBATCH --account=rpp-doesburg
#SBATCH --time=00:05:00
#SBATCH --job-name=eegfh
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G	# Actually 80 Gb of RAM was used for step 'add_brain_noise'
				# when BEM and FS were already precalculated
#SBATCH --mail-user=amoiseev@sfu.ca

module load python/3.8.10
module load scipy-stack

cd  /home/amoiseev/projects/rpp-doesburg/amoiseev/mycode/python/eegfhabrainage
source mne/bin/activate
cd preprocessing

python test_preproc.py
deactivate

