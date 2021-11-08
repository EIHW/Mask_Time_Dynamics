import pickle as p
import socket
import torch
import torch.nn as nn
# from pytorch_wavelets import DWTForward
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pywt
import numpy as np
import librosa
from torchaudio.transforms import Spectrogram, MelSpectrogram, MFCC, AmplitudeToDB, ComputeDeltas
from pyAudioAnalysis.ShortTermFeatures import feature_extraction, speed_feature
from pyAudioAnalysis.audioBasicIO import read_audio_file as wavread
from scipy import stats


def waveletPacket(data, n=3, if_print=False, wavelet='db1'):
    data = data.tolist()
    result = []
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=n)
    for node in wp.get_level(n, 'freq'):
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=n)
        name = node.path
        new_wp[name] = wp[name].data
        ans = new_wp.reconstruct(update=False)
        ans = ans[:len(data)]
        result.append(ans)
        if if_print:
            print((name, ans))
    result = np.array(result).reshape(2**n, len(data)).astype(np.float32)
    return result


def listen_wav(path, wav, sr):
    torchaudio.save(path, wav, sr)


class DataETL(Dataset):
    def __init__(self, seeds_path, mother_wav=None, depth=3, lld=False, agwn=False, distributed=False, gender='both', features='MDD'):
        super(DataETL, self).__init__()
        with open(seeds_path, 'rb') as pkl_file:
            self.seeds = p.load(pkl_file)

        if socket.gethostname() == 'UAU-86505':
            machine = '/home/user/on_gpu/'
        else:
            machine = '/home/liushuo/'

        with open(machine + '/Mask_Supervised_WaveNet_AutoEncoder/seeds/gender_dict.pkl', 'rb') as gender_pkl:
            self.gender_dict = p.load(gender_pkl)

        self.agwn = agwn
        self.distributed = distributed

        self.label2_num = {'clear': 0,
                           'mask': 1}

        self.gender2_num = {'m': 0,
                            'f': 1}

        if gender == 'both':
            self.normalise = {
                'feature': {
                    'mean': -13.9932,
                    'std': 17.8872
                },
                'deltas': {
                    'mean': 0.0117,
                    'std': 1.6170
                },
                'deltas_deltas': {
                    'mean': -0.00025,
                    'std': 0.60763
                }
            }
        elif gender == 'female':
            self.normalise = {
                'feature': {
                    'mean': -12.7368,
                    'std': 17.3681
                },
                'deltas': {
                    'mean': 0.0176,
                    'std': 1.7054
                },
                'deltas_deltas': {
                    'mean': 0.0002,
                    'std': 0.62905
                }
            }
        else:
            self.normalise = {
                'feature': {
                    'mean': -14.8783,
                    'std': 18.2368
                },
                'deltas': {
                    'mean': 0.0079,
                    'std': 1.5704
                },
                'deltas_deltas': {
                    'mean': -0.0005,
                    'std': 0.5969
                }
            }

        self.mother_wav = mother_wav
        self.depth = depth
        self.lld = lld

        self.features = features
        self.mel_spec = MelSpectrogram(sample_rate=16000, n_mels=40, hop_length=50)
        self.amp2db = AmplitudeToDB()
        self.feature_converter = nn.Sequential(self.mel_spec, self.amp2db)
        self.get_deltas = ComputeDeltas()

    def __getitem__(self, idx):
        seed = self.seeds[idx]
        seed_path, label = seed[0], self.label2_num[seed[1]]
        waveform, fs = torchaudio.load(seed_path)

        if self.agwn is True:
            waveform = self.addAWGN(waveform)

        feature = (self.feature_converter(waveform) - self.normalise['feature']['mean']) / self.normalise['feature']['std']
        deltas = (self.get_deltas(feature) - self.normalise['deltas']['mean']) / self.normalise['feature']['std']
        deltas_deltas = (self.get_deltas(deltas) - self.normalise['deltas_deltas']['mean']) / self.normalise['feature']['std']

        if self.features == 'MDD':
            features = torch.cat([feature, deltas, deltas_deltas], dim=0)   # bs, ch, F, T
        elif self.features == 'MD':
            features = torch.cat([feature, deltas], dim=0)   # bs, ch, F, T
        elif self.features == 'M':
            features = feature   # bs, ch, F, T

        if self.distributed:
            features = features.unfold(-1, 8, 4).permute(2, 0, 1, 3)

        if self.lld:
            lld_features = self.extract_LLDs(seed_path)
            return features, torch.from_numpy(lld_features).float(), label

        if self.mother_wav is not None:
            wavelets = waveletPacket(waveform, n=self.depth, wavelet=self.mother_wav)
            return torch.from_numpy(wavelets), label
        elif self.mother_wav == 'orig':
            return torch.from_numpy(waveform).unsqueeze(-2), label
        else:
            return features, label, self.gender2_num[self.gender_dict[seed_path.split('/')[-1]]]

    def __len__(self):
        return len(self.seeds)

    def extract_LLDs(self, seed_path):
        fs, wavform = wavread(seed_path)
        features, f_names = feature_extraction(wavform, fs, 400, 50)  # F1 x T
        speech_features = speed_feature(wavform, fs, 400, 50).T    # F2 x T
        features = np.concatenate([features, speech_features], axis=0)          # (F1 + F2) x T
        features = stats.zscore(features, axis=1)
        return features

    def addAWGN(self, signal, num_bits=16, snr_low=25, snr_high=30):
        # Generate White Gaussian noise
        noise = torch.randn_like(signal)
        signal_len = signal.shape[1]
        # Normalize signal and noise
        norm_constant = 2.0 ** (num_bits - 1)
        signal_norm = signal / norm_constant
        noise_norm = noise / norm_constant
        # Compute signal and noise power
        s_power = torch.sum(signal_norm ** 2, dim=1) / signal_len
        n_power = torch.sum(noise_norm ** 2, dim=1) / signal_len
        # Random SNR: Uniform [15, 30] in dB
        target_snr = torch.randint(low=snr_low, high=snr_high, size=(1, ))[0]
        # Compute K (covariance matrix) for each noise
        K = torch.sqrt((s_power / n_power) * 10 ** (- target_snr / 10.))
        K = torch.ones_like(signal) * K
        # Generate noisy signal
        return signal + K * noise


if __name__ == '__main__':
    seeds_path = './seeds/' + 'test.pkl'
    dataset = DataETL(seeds_path, mother_wav=None, lld=False, distributed=False)
    dl = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=16)

    for idx, (batch, labels, genders) in enumerate(dl, 1):
        print('[{}]: {} - {} - {}'.format(idx, batch.shape, labels.shape, genders.shape))
