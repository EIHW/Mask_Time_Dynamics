# This file is used to compute mean and standard deviation for original feature maps. their deltas and delta-deltas for
# feature normalisation.

import pickle as p
import numpy as np
import torchaudio
from torchaudio.transforms import Spectrogram, MelSpectrogram, MFCC, AmplitudeToDB, ComputeDeltas
from tqdm import tqdm
from pyAudioAnalysis.ShortTermFeatures import feature_extraction, speed_feature
from pyAudioAnalysis.audioBasicIO import read_audio_file as wavread
from scipy import stats


def get_statistics(pkls, content):
    seeds_all = []
    for pkl in pkls:
        with open(pkl, 'rb') as p_file:
            seeds = p.load(p_file)
            seeds_all.extend(seeds)

    mel_spec = MelSpectrogram(sample_rate=16000, n_mels=40, hop_length=50)
    amp2db = AmplitudeToDB()
    get_deltas = ComputeDeltas()

    results = {
        'feature': {},
        'deltas': {},
        'deltas_deltas': {}
    }

    feature_values = []
    deltas_values = []
    deltas_deltas_values = []
    for seed in tqdm(seeds_all):
        path = seed[0]
        waveform, fs = torchaudio.load(path)
        feature = mel_spec(waveform)
        feature = amp2db(feature)
        deltas = get_deltas(feature)
        deltas_deltas = get_deltas(deltas)
        if content == 'feature':
            feature = feature.view(-1).detach().numpy()
            feature_values.append(feature)
        elif content == 'deltas':
            deltas = deltas.view(-1).detach().numpy()
            deltas_values.append(deltas)
        elif content == 'deltas_deltas':
            deltas_deltas = deltas_deltas.view(-1).detach().numpy()
            deltas_deltas_values.append(deltas_deltas)

    if content == 'feature':
        feature_all = np.concatenate(feature_values)
        feature_mean = np.mean(feature_all)
        feature_std = np.std(feature_all)
        results['feature']['mean'] = feature_mean
        results['feature']['std'] = feature_std
    elif content == 'deltas':
        deltas_all = np.concatenate(deltas_values)
        deltas_mean = np.mean(deltas_all)
        deltas_std = np.std(deltas_all)
        results['deltas']['mean'] = deltas_mean
        results['deltas']['std'] = deltas_std
    elif content == 'deltas_deltas':
        deltas_deltas_all = np.concatenate(deltas_deltas_values)
        deltas_deltas_mean = np.mean(deltas_deltas_all)
        deltas_deltas_std = np.std(deltas_deltas_all)
        results['deltas_deltas']['mean'] = deltas_deltas_mean
        results['deltas_deltas']['std'] = deltas_deltas_std

    return results


def extract_LLDs(seed_path):
    fs, wavform = wavread(seed_path[0])
    features, f_names = feature_extraction(wavform, fs, 400, 50)  # F1 x T
    speed_features = speed_feature(wavform, fs, 400, 50)    # F2 x T
    features = np.append(features, speed_features)          # (F1 + F2) x T
    features = stats.zscore(features, axis=0)
    return features


def lld_statistics(pkls):
    seeds_all = []
    for pkl in pkls:
        with open(pkl, 'rb') as p_file:
            seeds = p.load(p_file)
            seeds_all.extend(seeds)

    for seed in seeds_all:
        extract_LLDs(seed)


if __name__ == '__main__':
    pkl_files = [
        './seeds/train_valid_sel.pkl',
        './seeds/valid.pkl',
        './seeds/test.pkl',
    ]

    result = get_statistics(pkl_files, 'feature')
    print(result)
    result = get_statistics(pkl_files, 'deltas')
    print(result)
    result = get_statistics(pkl_files, 'deltas_deltas')
    print(result)

