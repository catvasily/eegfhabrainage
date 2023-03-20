def get_data_folders():
	'''Setup input and output data folders depending on the host machine.

	Returns:
	    data_root, out_root (str, str): paths to the root input and output
		data folders
	    
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
	elif 'cedar' in host:
		data_root = '/project/6019337/databases/eeg_fha/release_001/edf'
		out_root = user_home + '/projects/rpp-doesburg/' + user + '/data/eegfhabrainage/processed'
	else:
		home_dir = os.getcwd()
		data_root = home_dir
		out_root = home_dir + '/processed'

	return data_root, out_root

"""A test script
based on the example code in \":doc:`README`\" section.

"""
# The 'if' is needed to prevent running this code when the file is 
# imported into some other source and is not supposed to run
if __name__ == '__main__': 
	# Test processing a single record using the PreProcessing class directly
	from edf_preprocessing import PreProcessing

	# Inputs
	hospital = 'Burnaby'	# Either Burnaby or Abbotsford
	bRunSingleFile = False	# Flag to test single PreProcessing class instance

	data_root, out_root = get_data_folders()

	# Burnaby
	file_name = "81c0c60a-8fcc-4aae-beed-87931e582c45.edf"
	#file_name = "81be60fc-ed17-4f91-a265-c8a9f1770517.edf"
	#file_name = "fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2.edf"
	#file_name = "ffff1021-f5ba-49a9-a588-1c4778fb38d3.edf"
	#source_scan_ids = ["fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2", "ffff1021-f5ba-49a9-a588-1c4778fb38d3"]
	source_scan_ids = None

	# Abbotsford
	#file_name = "fffcedd0-503f-4400-8557-f74b58cff9d9.edf"	# HV
	#file_name = "fffaab93-e908-4b93-a021-ab580e573585.edf"	# No HV
	#file_name = "test_HV.edf"				# Several HV series

	input_dir = data_root + "/" + hospital
	output_dir = out_root + "/" + hospital
	path = input_dir + '/' + file_name

	# These will be used if explicitly specified in the PreProcessing class constructor
	target_frequency=200
	lfreq=0.5
	hfreq=55

	if bRunSingleFile:
		# Initiate the preprocessing object, filter the data between 0.5 Hz and 55 Hz and resample to 200 Hz.
		# p = PreProcessing(path, target_frequency=target_frequency, lfreq=lfreq, hfreq=hfreq)
		p = PreProcessing(path)		# All parms will be read from the JSON conf file

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

		print('\nDone processing a single file {} using the PreProcessing class.'.format(file_name))

	# Test processing multiple records using slice_edfs() function
	from edf_preprocessing import slice_edfs

	slice_edfs(input_dir, output_dir, source_scan_ids = source_scan_ids)

