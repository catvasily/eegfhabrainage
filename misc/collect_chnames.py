import os
import mne

# data_root = "/data/eegfhabrainage"
data_root = "/project/6019337/databases/eeg_fha/release_001/edf/"	# cedar
hospitals = ['Abbotsford', 'Burnaby']

file_type = ".edf" # replace with the f extension you want to find
all_chans = set()
hosp_chans = []
cnt = 0
max_cnt = 100
stop = False

for hosp in hospitals:
    if stop:
        break;

    folder_path = os.path.join(data_root, hosp)
    hosp_set = set()

    for f in os.listdir(folder_path):
        fpath = os.path.join(folder_path, f)

        if f.endswith(file_type) and not os.path.isdir(fpath):
            try:
                raw = mne.io.read_raw_edf(fpath, preload=False, verbose = False)
                s = set(raw.ch_names)
                hosp_set |= s
                all_chans |= s
                cnt += 1

                if cnt == max_cnt:
                    stop = True
                    break
            except Exception as e:
                print('Record {} !!! FAILED !!!'.format(hosp + '/' + f))
                print(e, flush = True)

    hosp_chans.append(hosp_set)

for i,hosp in enumerate(hosp_chans):
    print(hospitals[i])
    l = list(hosp)
    l.sort()
    print(l)

lst_chans = list(all_chans)
lst_chans.sort()

print('\nAll channels:')
print(lst_chans)

