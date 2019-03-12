import numpy as np
import librosa
from librosa.feature import melspectrogram
import pandas as pd
import soundfile
import argparse
import yaml
import os
import h5py
from utils.preprocessing import convert2mel, normalize_amplitude, windowing



####### ARGUMENTS (JUST THE PATH FOR NOW) #######
#### we have to normalize the amplitude and also the spectrum!! because of the microphones!!

"""
Train entries: 17585
New train entries with some space for validationL: 17310 -> 271 batches
val set: 275 -> 5 batches
Test entries: 947 -> 15 batches
Obs: we will start working with batches using chunking
"""
parser = argparse.ArgumentParser(description='Code for DCASE Challenge task 2.')
parser.add_argument('-s','--directory',dest='directory',action='store',required=False, type=str, default='/home/fabian/project_audio/')
parser.add_argument('-t','--type_trainig',dest='type_training',action='store',required=True, type=str)
parser.add_argument('-a','--audio_dir',dest='audio_dir',action='store',required=False, type=str,default='home/fabian/DataProcessed/')


args = parser.parse_args()
type_training= args.type_training
path_to_metadata =  args.directory +'FSDnoisy18k.meta/' + type_training + '_set.csv'
hdf5_name = "processed_data_" + type_training  +  ".hdf5"
audio_path= args.audio_dir + "processed_wavs_" + type_training
if type_training != 'train':
    audio_path = args.audio_dir

df_train = pd.read_csv(path_to_metadata)
fname = df_train['fname'].values

n_mels = 96
fs= 32000 # we will make downsampling to save some data!!44100
n_fft = 2048
windows_size_s = 35 # 30 milisecons windowing (to have more context)
windows_size_f = (windows_size_s * fs ) // 1000  # int division # 960 samples
hop_length_samples = int(windows_size_f // 2) ## 480 samples
audio_duration_s = 2  # 2 seconds
audio_duration = audio_duration_s * 1000
number_of_frames = fs * audio_duration # deprecated, use short audio in database already
fmax = int(fs / 2)
fmin = 0
normalize_audio = True
spectrogram_type = 'power'



hdf5_store = h5py.File(hdf5_name, "w")
all_inputs = hdf5_store.create_dataset("all_inputs" , (len(df_train['fname'].values),1,n_mels,115), compression="gzip")
dt = h5py.special_dtype(vlen=str)
targets = hdf5_store.create_dataset("targets", data = df_train['label'].values, dtype=dt ,compression="gzip")
data_name = hdf5_store.create_dataset("audio_name",data = fname, dtype=dt , compression='gzip')
if type_training == 'train':
    manually_verified = hdf5_store.create_dataset("manually_verified", dtype='i1' ,data = df_train['manually_verified'].values, compression="gzip")
    noisy_small =  hdf5_store.create_dataset("noisy_small", dtype='i1' ,data = df_train['noisy_small'].values, compression="gzip")

data_processed = [convert2mel(audio,audio_path,fs, n_fft,fmax,n_mels,hop_length_samples, windows_size_f,type_training) for ii,audio in enumerate(fname)]
hdf5_store['all_inputs'][:,0] = data_processed
print("saving data for experiment")
hdf5_store.close()




