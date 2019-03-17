import numpy as np
import librosa
from librosa.feature import melspectrogram
from librosa.core import stft
import soundfile
import os
import scipy

"""
Functions for the pre processing

"""

def convert2mel(audio,base_path,fs, n_fft,fmax,n_mels,hop_length_samples, window_lenght,type_training, data_augmentation):
    """
    Convert raw audio to mel spectrogram
    """
    path = os.path.join(base_path, audio + "-" + data_augmentation.split("_")[0] + ".wav")

    if type_training != "train":
        if os.path.isfile(os.path.join(base_path,"data_train_"+ data_augmentation,audio+ "-"+ data_augmentation.split("_")[0]+ ".wav")):
            data,_ = librosa.core.load(os.path.join(base_path,"data_train_"+ data_augmentation,audio+ "-"+ data_augmentation.split("_")[0]+ ".wav"), sr=fs, res_type="kaiser_best")
        else:
            data,_ = librosa.core.load(os.path.join(base_path,"data_test_"+ data_augmentation,audio+"-"+data_augmentation.split("_")[0]+".wav"), sr=fs, res_type="kaiser_best")
    else:
        data, _ = librosa.core.load(path, sr=fs, res_type="kaiser_best")
    if data.shape[0] < 64000:
        data = np.pad(data, (( 64000 - data.shape[0])//2 +1, ( 64000 - data.shape[0])//2 +1 )   , 'constant', constant_values=(0,0))
    data = normalize_amplitude(data)[0:64000]
    powSpectrum = np.abs(stft(data+ 0.00001,n_fft,hop_length = hop_length_samples, win_length = window_lenght, window = windowing(window_lenght, sym=False), center=True, pad_mode='reflect'))**2

    mels = melspectrogram(y= None,n_fft=n_fft ,sr=fs ,S= powSpectrum, hop_length= hop_length_samples ,n_mels=n_mels,fmax=fmax , fmin = 0.0).T
    mels = librosa.core.power_to_db(mels, ref=np.min(mels))
    mels = mels / np.max(mels)
    return mels.T

##### Amplitude Normalization of audios #########

def normalize_amplitude(y, tolerance=0.005):

    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + tolerance
    return y / max_value

processes = []

##### SHORT TIME FOURIER TRANSFORM #############

def windowing(win_leng_frames, sym=False):
    return scipy.signal.hamming(win_leng_frames, sym=False)

### printing, deprecated function #######

def imprimir(ii,audio,base_path,fs, n_fft,fmax,n_mels,number_of_frames):
    mel = convert2mel(audio,base_path,fs, n_fft,fmax,n_mels,number_of_frames)
    if ii == 0:
        print(mel)
    return mel

####
