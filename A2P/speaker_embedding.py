import torch
import os
import torch.nn as nn
import numpy as np
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from collections import OrderedDict

class D_VECTOR(nn.Module):
    """d vector speaker embedding."""
    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        super(D_VECTOR, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell, 
                            num_layers=num_layers, batch_first=True)  
        self.embedding = nn.Linear(dim_cell, dim_emb)
        
        
    def forward(self, x):
        self.lstm.flatten_parameters()            
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:,-1,:])
        norm = embeds.norm(p=2, dim=-1, keepdim=True) 
        embeds_normalized = embeds.div(norm)
        return embeds_normalized

def load_embd(batchsize,speaker='shelly',dataroot='/content/Gestures'):
    path = os.path.join(dataroot,speaker+'_embd.npz')
    emb = np.load(path)
    emb = emb['embd']
    emb = torch.from_numpy(emb)
    emb = emb[None,:]
    emb = emb.expand(batchsize,emb.size(1))
    return emb


class GetEmbedding(object):
    def __init__(self,speaker='shelly',dataroot='/content/Gestures',num_uttrs=10,len_crop=128):
        sub_path = 'train/npz'
        self.speaker = speaker
        self.dataroot = dataroot
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_uttrs = num_uttrs
        self.len_crop = len_crop
        self.audio_path = os.path.join(dataroot,speaker,sub_path)
        self.b, self.a = self.butter_highpass(30, 16000, order=5)
        self.mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
        self.min_level = np.exp(-100 / 20 * np.log(10))

        self.C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval()
        c_checkpoint = torch.load('3000000-BL.ckpt',map_location=self.device)
        new_state_dict = OrderedDict()
        for key, val in c_checkpoint['model_b'].items():
            new_key = key[7:]
            new_state_dict[new_key] = val
        self.C.load_state_dict(new_state_dict)
        self.C.to(self.device)
        _, _, self.fileList = next(os.walk(self.audio_path))
        print('audio_nums = ',len(self.fileList))

    def save_embd(self):
        idx_uttrs = np.random.choice(len(self.fileList), size=self.num_uttrs, replace=False)
        embs = []
        for i in range(self.num_uttrs):
            tmp = np.load(os.path.join(self.audio_path, self.fileList[idx_uttrs[i]]))
            tmp = tmp['audio']
            tmp = self.get_spectrogram(tmp)

            if tmp.shape[0] < self.len_crop:
                print('shit!')

            left = np.random.randint(0, tmp.shape[0]-self.len_crop)
            melsp = torch.from_numpy(tmp[np.newaxis, left:left+self.len_crop, :]).to(self.device)
            emb = self.C(melsp)
            embs.append(emb.detach().squeeze().cpu().numpy())

        embs = np.mean(embs, axis=0)
        np.savez(self.dataroot+'/'+self.speaker+'_embd.npz',embd=embs)


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