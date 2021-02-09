import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import cv2
import random
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel


class Audio2GestureDataset(data.Dataset):
  def __init__(self,opt):
    super(Audio2GestureDataset, self).__init__()
    self.opt = opt
    self.data = pd.read_csv(os.path.join(opt.dataroot,"{}.csv".format(opt.phase)))

    self.b, self.a = self.butter_highpass(30, 16000, order=5)
    self.mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    self.min_level = np.exp(-100 / 20 * np.log(10))

  def __getitem__(self, index):
    row = np.load(self.data.iloc[index]['pose_fn'])
    pose = row['pose']
    pose = np.delete(pose,[7,8,9],axis=2)
    pose = np.delete(pose,[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],axis=2)
    audio = row['audio']
    audio = self.get_spectrogram(audio)

    if self.opt.pose_mode == 'relative':
        pass
    else:
        pose = pose/320.0
    
    pose = np.transpose(pose,(1,0,2))
    return {'pose':torch.Tensor(pose),'audio':audio}

  def get_spectrogram(self,audio):
    y = signal.filtfilt(self.b, self.a, audio)
    wav = y * 0.96 + (np.random.rand(y.shape[0])-0.5)*1e-06
    D = self.pySTFT(wav).T
    D_mel = np.dot(D, self.mel_basis)
    D_db = 20 * np.log10(np.maximum(self.min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)
    S = S.astype(np.float32)
    return S

  def butter_highpass(self, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
      
      
  def pySTFT(self,x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)

  def __len__(self):
    return len(self.data)