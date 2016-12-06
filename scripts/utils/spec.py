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
import progressbar
import gc
import h5py
#from joblib import Parallel,delayed
#import multiprocessing


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
    gram = librosa.feature.melspectrogram(y=speech,sr=16000,n_mels=256,n_fft=1024,hop_length=160)
    C_dash = 10000
    gram = np.log(1+C_dash*gram)
    length = gram.shape[1]
    no_of_frames = length/100
    if no_of_frames!=0:
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
    else :
        start = 0;
        end = length;
        gram = gram[:,start:end]
    return gram

def sounds_spkr(sounds):
    #sounds contains list of wav files for each speaker
    #helps in parallelizing better
    for sound in sounds:
        specgram(sound)


path_to_timit = "../../datasets/timit/"  #HACK:0 PATH BEWARE!! #TODO:0 FIX PATH ERRORS
#train_dialect = "train/dr1/"
dialects_train = glob.glob(path_to_timit+"train/*")
dialects_test = glob.glob(path_to_timit+"test/*")
dialects = dialects_train+dialects_test
speakers_dir = glob.glob(path_to_timit+'**/**/*')
speakers_ID = []
n_s = 0;
speakers = {}
for i in range(len(speakers_dir)):
    (bb,spk) = os.path.split(speakers_dir[i])
    speakers_ID.append(spk)
    speakers[spk] = n_s
    n_s = n_s+1

#BOY OH BOY 630 speakers
total_folders = len(speakers_dir);
bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
],max_value=total_folders)
## This hack couldn't be avoided, apologies
#Creating the dataset, takes a while ~ 1hr
for i in xrange(len(speakers_dir)):
    speaker_path = speakers_dir[i]
    sounds = glob.glob(speaker_path+'/*.wav')
    for sound in sounds:
        gram = specgram(sound);
    if i%100==0:
        gc.collect()
    bar.update(i)
bar.finish()

'''
#Parallel version of the above loop
num_cores = multiprocessing.cpu_count()

parallelizer = Parallel(n_jobs=num_cores)
task_iterator = (delayed(sounds_spkr) (glob.glob(speaker_path+'/*.wav')) for speaker_path in speakers_dir)

#^^ should give a massive performance boost.
'''


#datasets created
#we will move these into a separate directory called ../datasets/train_set/speaker_name
train_set = glob2.glob(path_to_timit+"**/*.png")
for fil in train_set:
    f_name,speaker_name = path_info(fil)
    destn = '../../datasets/train_set/'+speaker_name+'_'+f_name
    shutil.copy2(src=fil,dst=destn)
#making a pickle out of all these pictures in ../datasets/train_set
train_set = glob2.glob('../../datasets/train_set/*.png')
train = np.zeros((1,256*100+1))
p1 = re.compile("([a-z0-9A-Z]+)_([a-z0-9A-Z]+)_(?:[0-9]).png")
bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
],max_value=len(train_set))

for i in xrange(len(train_set)):
    if i%2==0 :
        continue
    if i%100==0 :
        gc.collect()
    image = train_set[i]
    (file_name,temp) = path_info(image)
    m1 = p1.match(string=file_name)
    (speaker_name,temp) = m1.groups(0)
    img = (scipy.misc.imread(name=image,flatten=True))/255.0
    img = np.reshape(a=img,newshape=(1,-1))
    spkr = speakers[speaker_name]
    img = np.append(img,spkr)
    train = np.vstack((train,img))
    bar.update(i)
train  = train[1:,:]
images = train[:,:-1];
labels = train[:,-1];
with open('../cnn_model/train_data.pkl','wb') as f1:
    pkl.dump(train,f1)
shutil.rmtree('../../datasets/train_set'); #clearing the residues
os.system('mkdir ../../datasets/train_set')

with h5py.File('../cnn_model/train.h5','w') as f1:
    dset_img = f1.create_dataset("imgs",data=images)
    dset_lbl = f1.create_dataset("lbls",data=labels)
