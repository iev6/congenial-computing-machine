'''
Spectrogram of wav files in TIMIT dataset

@author : Giridhur S
'''
import os
import wave as wav
import numpy as np
import scipy
import librosa.feature
import glob
import glob2
import shutil

def path_info(audio_path):
    (dir_name,file_name) = os.path.split(audio_path)
    (dir_name,speaker_name) = os.path.split(dir_name)
    return file_name,speaker_name

def specgram(audio_path='test.wav',params=None):

    if params is not None:
        n_fft = params.n_fft
        hop_length = params.hop_length
        window = params.window
        n_mels = params.n_mels
        center = False
    else :
        n_fft = 1024
        hop_length = 160
        center = False
        n_mels = 128
    (speech,rate) = librosa.load(path=audio_path,sr=16000) #16Khz sampling rate #HACK: HARDCODED
    gram = librosa.feature.melspectrogram(y=speech,sr=16000,n_mels=128,n_fft=1024,hop_length=160)
    C_dash = 10000
    gram = np.log(1+C_dash*gram)
    length = gram.shape[1]
    no_of_frames = length/100
    start = (length%100)/2
    if (length%100)%2==0:
        end = length-(length%100)/2
    else :
        end = length-(length%100)/2-1
    gram_tf = gram[:,start:end]
    grams = np.hsplit(indices_or_sections=no_of_frames,ary=gram_tf)
    for i in xrange(0,no_of_frames):
        img_file =audio_path.rstrip('.wav')+'_'+str(i)+'.png'
        scipy.misc.imsave(img_file,grams[i])
    return gram


path_to_timit = "../../datasets/timit/"  #HACK:0 PATH BEWARE!! #TODO:0 FIX PATH ERRORS
train_dialect = "train/dr1/"
test_dialect = "test/dr1/"
speakers = {"fcjf0":1,"fdaw0":2,"fdml0":3,"fecd0":4}

#Creating the datasets
for speaker in speakers.keys():
    speaker_path = path_to_timit+train_dialect+speaker+"/*.wav"
    sounds = glob.glob(speaker_path)
    for sound in sounds:
        gram = specgram(sound);
#datasets created
#we will move these into a separate directory called ../datasets/train_set
train_set = glob2.glob(path_to_timit+"**/*.png")
for fil in train_set:
    destn = '../../datasets/train_set'
    shutil.copy2(src=fil,dst=destn)
#making a pickle out of all these pictures in ../datasets/train_set
