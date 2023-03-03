"""A test script
based on the example code in \":doc:`README`\" section.
"""
# The 'if' is needed to prevent running this code when the file is 
# imported into some other source and is not supposed to run
if __name__ == '__main__': 
	# Test processing a single record using the PreProcessing class directly
	from edf_preprocessing import PreProcessing

	file_name = "81c0c60a-8fcc-4aae-beed-87931e582c45.edf"
	#file_name = "81be60fc-ed17-4f91-a265-c8a9f1770517.edf"
	#file_name = "fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2.edf"
	#file_name = "ffff1021-f5ba-49a9-a588-1c4778fb38d3.edf"

	input_dir = "/data/eegfhabrainage"
	path = input_dir + '/' + file_name
	output_dir = "/data/eegfhabrainage/results"
	target_frequency=200
	lfreq=0.5
	hfreq=55
	target_length=60
	target_segments=5

	# Initiate the preprocessing object, filter the data between 0.5 Hz and 55 Hz and resample to 200 Hz.
	p = PreProcessing(path, target_frequency=target_frequency, lfreq=lfreq, hfreq=hfreq)

	# This calls internal functions to detect 'bad intervals' and define 5 'good' ones 60 seconds each
	p.extract_good(target_length=target_length, target_segments=target_segments)

	# Calling the function saves new EDF files to output_folder. In case there are more than 1, it adds suffix "_n" to the file name 
	p.save_edf(folder=output_dir, filename=file_name)

	# Extract and convert data to Numpy arrays
	p.create_intervals_data()
	df = p.intervals_df

	print('\nDone processing a single file {} using the PreProcessing class.'.format(file_name))

	# Test processing multiple records using slice_edfs() function
	from edf_preprocessing import slice_edfs

	source_scan_ids = ["fff0b7a0-85d6-4c7e-97be-8ae5b2d589c2", "ffff1021-f5ba-49a9-a588-1c4778fb38d3"]
	slice_edfs(input_dir, output_dir, target_length, source_scan_ids = source_scan_ids,
               target_segments=target_segments, nfiles=None)

