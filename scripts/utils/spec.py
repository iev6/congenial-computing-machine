'''
Spectrogram of wav files in TIMIT dataset

@author : Giridhur S, Ramasubramanian B
'''
import os
import wave as wav
import numpy as np
import scipy
import librosa.feature
import glob
import glob2
import shutil
import cPickle as pkl
import re

def path_info(audio_path):
    (dir_name,file_name) = os.path.split(audio_path)
    (dir_name,speaker_name) = os.path.split(dir_name)
    return file_name,speaker_name

def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

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
speakers = {'fcjf0': 21,
 'fdaw0': 28,
 'fdml0': 33,
 'fecd0': 5,
 'fetb0': 9,
 'fjsp0': 36,
 'fkfb0': 23,
 'fmem0': 27,
 'fsah0': 25,
 'fsjk1': 26,
 'fsma0': 34,
 'ftbr0': 2,
 'fvfb0': 14,
 'fvmh0': 18,
 'mcpm0': 17,
 'mdac0': 19,
 'mdpk0': 16,
 'medr0': 38,
 'mgrl0': 22,
 'mjeb1': 29,
 'mjwt0': 30,
 'mkls0': 35,
 'mklw0': 31,
 'mmgg0': 12,
 'mmrp0': 7,
 'mpgh0': 3,
 'mpgr0': 11,
 'mpsw0': 10,
 'mrai0': 6,
 'mrcg0': 15,
 'mrdd0': 20,
 'mrso0': 1,
 'mrws0': 8,
 'mtjs0': 13,
 'mtpf0': 4,
 'mtrr0': 32,
 'mwad0': 37,
 'mwar0': 24}

#Creating the datasets
for speaker in speakers.keys():
    speaker_path = path_to_timit+train_dialect+speaker+"/*.wav"
    sounds = glob.glob(speaker_path)
    for sound in sounds:
        gram = specgram(sound);
#datasets created
#we will move these into a separate directory called ../datasets/train_set/speaker_name
train_set = glob2.glob(path_to_timit+"**/*.png")
for fil in train_set:
    f_name,speaker_name = path_info(fil)
    destn = '../../datasets/train_set/'+speaker_name+'_'+f_name
    shutil.copy2(src=fil,dst=destn)
#making a pickle out of all these pictures in ../datasets/train_set
train_set = glob2.glob('../../datasets/train_set/*.png')
train = np.zeros((1,128*100+1))
p1 = re.compile("([a-z0-9A-Z]+)_([a-z0-9A-Z]+)_(?:[0-9]).png")
for image in train_set:
    (file_name,temp) = path_info(image)
    m1 = p1.match(string=file_name)
    (speaker_name,temp) = m1.groups(0)
    img = (scipy.misc.imread(name=image,flatten=True))/255.0
    img = np.reshape(a=img,newshape=(1,-1))
    spkr = speakers[speaker_name]
    img = np.append(img,spkr)
    train = np.vstack((train,img))
train  = train[1:,:]
with open('../cnn_model/train_data.pkl','wb') as f1:
    pkl.dump(train,f1)
