def get_data_folders():
	'''Setup input and output data folders depending on the host machine.

	Returns:
	    data_root, out_root, cluster_job (str, str, bool): paths to the root input and output
		data folders, and a flag indicating whether host is on CC cluster
	    
	'''
	import os.path as path
	import socket

	host = socket.gethostname()

	# path.expanduser("~") results in /home/<username>
	user_home = path.expanduser("~")
	user = path.basename(user_home) # Yields just <username>

	if 'ub20-04' in host:
		data_root = '/data/eegfhabrainage'
		out_root = data_root + '/processed'
		cluster_job = False
	elif 'cedar' in host:
		data_root = '/project/6019337/databases/eeg_fha/release_001/edf'
		out_root = user_home + '/projects/rpp-doesburg/' + user + '/data/eegfhabrainage/processed'
		cluster_job = True
	else:
		home_dir = os.getcwd()
		data_root = home_dir
		out_root = home_dir + '/processed'
		cluster_job = False

	return data_root, out_root, cluster_job

"""A test script
based on the example code in \":doc:`README`\" section.

"""
# The 'if' is needed to prevent running this code when the file is 
# imported into some other source and is not supposed to run
if __name__ == '__main__': 
	# Test processing a single record using the PreProcessing class directly
	import sys
	import glob
	import os.path as path
	from edf_preprocessing import PreProcessing

	# Inputs
	N_ARRAY_JOBS = 100	# Number of parrallel jobs to run on cluster
	hospital = 'Burnaby'	# Either Burnaby or Abbotsford
	bRunSingleFile = False	# Flag to test single PreProcessing class instance

	data_root, out_root, cluster_job = get_data_folders()

	# Burnaby
	file_name = "81c0c60a-8fcc-4aae-beed-87931e582c45.edf"
	#file_name = "81be60fc-ed17-4f91-a265-c8a9f1770517.edf"
	#file_name = "fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2.edf"
	#file_name = "ffff1021-f5ba-49a9-a588-1c4778fb38d3.edf"
	#source_scan_ids = ["fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2", "ffff1021-f5ba-49a9-a588-1c4778fb38d3"]
	#source_scan_ids = [
        #        '819ebadf-1bcb-4c35-8280-fb63d4747b35','ffedacda-ce90-452c-8007-f46ec1a04cd1',
	#	'81a91d9d-281d-4321-a656-5b68ecb37090','ffee84ea-1238-4ef5-99fd-8ea9a05b98ca',
	#	'81aa06db-db61-45af-965d-71813cf34a81','ffef5962-ed51-45d6-b20a-a95dd1f6ddde',
	#	'81be60fc-ed17-4f91-a265-c8a9f1770517','fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2',
	#	'81c0c60a-8fcc-4aae-beed-87931e582c45','ffff1021-f5ba-49a9-a588-1c4778fb38d3']
	source_scan_ids = None

	# Abbotsford
	#file_name = "fffcedd0-503f-4400-8557-f74b58cff9d9.edf"	# HV
	#file_name = "fffaab93-e908-4b93-a021-ab580e573585.edf"	# No HV
	#file_name = "test_HV.edf"				# Several HV series

	input_dir = data_root + "/" + hospital
	output_dir = out_root + "/" + hospital
	pathname = input_dir + '/' + file_name

	# These will be used if explicitly specified in the PreProcessing class constructor
	target_frequency=200
	lfreq=0.5
	hfreq=55

	if bRunSingleFile:
		# Initiate the preprocessing object, filter the data between 0.5 Hz and 55 Hz and resample to 200 Hz.
		# p = PreProcessing(pathname, target_frequency=target_frequency, lfreq=lfreq, hfreq=hfreq)
		p = PreProcessing(pathname)		# All parms will be read from the JSON conf file

		# This calls internal functions to detect 'bad intervals' and define 5 'good' ones 60 seconds each
		# p.extract_good(target_length=target_length, target_segments=target_segments) QQQ
		p.extract_good()

		# import sys
		# sys.exit()	# QQQQQ

		# Calling the function saves new EDF files to output_folder. In case there are more than 1, it adds suffix "_n" to the file name 
		p.save_edf(folder=output_dir, filename=file_name)

		# Extract and convert data to Numpy arrays
		p.create_intervals_data()
		df = p.intervals_df

		print('\nDone processing a single file {} using the PreProcessing class.\n'.format(file_name))

	# Test processing multiple records using slice_edfs() function
	from edf_preprocessing import slice_edfs

	# When running on the CC cluster, 1st command line argument is a 0-based
	# array job index
	if len(sys.argv) == 1:	# No command line args
		ijob = 0
	else:
		ijob = int(sys.argv[1])

	if source_scan_ids is None:
		source_scan_ids = [path.basename(f)[:-4] for f in glob.glob(input_dir + '/*.edf')]

	if cluster_job:
		nfiles = len(source_scan_ids)
		files_per_job = nfiles // N_ARRAY_JOBS + 1
		istart = ijob * files_per_job

		if istart > nfiles - 1:
			print('All done')
			sys.exit()

		iend = min(istart + files_per_job, nfiles)
		source_scan_ids = source_scan_ids[istart:iend]

	slice_edfs(input_dir, output_dir, source_scan_ids = source_scan_ids)

