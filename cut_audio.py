import librosa
from os import listdir
from os.path import isfile, join
import IPython.display as ipd
import numpy as np
import soundfile

path_to_folder = "/home/fabian/Dropbox/FabianCloud/MscAI/SecondSemester/MLP Second Semester/PersonalRepo/real_data/FSDnoisy18k.audio_test"
count_number_of_files_changed = 0

for count, file in enumerate(listdir(path_to_folder)):
   if isfile(join(path_to_folder, file)):
       y, sr = librosa.load(path_to_folder + "/" + file,32000)
       yt, index = librosa.effects.trim(y)
       change = librosa.get_duration(y) - librosa.get_duration(yt)
       if change > 0.0:
           print(count, round(change, 3), "seconds")
           count_number_of_files_changed += 1
       soundfile.write('/home/fabian/DataProcessed/processed_wavs_test/' + file, yt[:sr*2], sr)
       if count%100 == 0:
           print(count, "/", len(listdir(path_to_folder)))
print(count_number_of_files_changed)


path_to_folder = "/home/fabian/DataProcessed/processed_wavs_test"
count_number_of_files_changed = 0

for count, file in enumerate(listdir(path_to_folder)):
   if isfile(join(path_to_folder, file)):
       print(count)
       f,e = librosa.load(path_to_folder + "/" + file,32000) #soundfile.read(path_to_folder + "\\" + file)
       if len(f) < e*2:
           print("HEREEEE", file)
           mul = int(round(2*e / len(f), 0)) + 1
           f = np.tile(f, mul).T

           if f.shape[0] > 2*e:
               f = np.delete(f, np.arange(2*e, f.shape[0]), axis=0)
       soundfile.write('/home/fabian/DataProcessed/processed_wavs_test/' + file, f, e)
