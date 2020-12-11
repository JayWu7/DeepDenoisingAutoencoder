#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as path
import scipy.signal as sig
import scipy.io.wavfile as wav
import os
import datetime
import pysepm

# 1.1. Read the audio file SX83.WAV and sampling rate
# data_root = '/Users/jaywu/Desktop/courses/Speech Recognition/Project/DDAE_1120'
data_root = '/Users/jaywu/Desktop/courses/Speech Recognition/Project/DDAE_1124'
Nfft = 1024
Fs_target = 16000


# sound_file = path.join(file_path, 'SX83.wav')

def get_matrix(sound_file):
    Fs, in_sig = wav.read(sound_file)  # Read audio file

    # 1.2. Make sure the sampling rate is 16kHz, resample if necessary
    Fs_target = 16000

    if Fs != Fs_target:
        in_sig = sig.resample_poly(in_sig, Fs_target, Fs)
        Fs = Fs_target

    return Fs, in_sig


def evaluate(result_file):
    reg_root = path.join(data_root, 'REG/')
    source_root = path.join(data_root, 'Source/')
    target_root = path.join(data_root, 'Target/')
    audios = os.listdir(reg_root)
    with open(result_file, 'a') as f:
        f.write('Evaluation results. time: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n')
        for audio in audios:
            reg_audio = path.join(reg_root, audio)
            source_audio = path.join(source_root, audio)
            target_audio = path.join(target_root, audio)

            fs, reg_matrix = get_matrix(reg_audio)
            _, source_matrix = get_matrix(source_audio)
            _, target_matrix = get_matrix(target_audio)

            enhanced_snrseg = pysepm.SNRseg(target_matrix, reg_matrix, fs)
            noisy_snrseg = pysepm.SNRseg(target_matrix, source_matrix, fs)

            enhanced_pesq = pysepm.pesq(target_matrix, reg_matrix, fs)
            noisy_pesq = pysepm.pesq(target_matrix, source_matrix, fs)

            enhanced_stoi = pysepm.stoi(target_matrix, reg_matrix, fs)
            noisy_stoi = pysepm.stoi(target_matrix, source_matrix, fs)

            f.write('Audio:{}:\n'.format(audio))
            f.write('SNRSEG for enhanced and noisy respectively: {}, {}\n'.format(enhanced_snrseg, noisy_snrseg))
            f.write('PESQ for enhanced and noisy respectively: {}, {}\n'.format(enhanced_pesq, noisy_pesq))
            f.write('STOI for enhanced and noisy respectively: {}, {}\n\n\n'.format(enhanced_stoi, noisy_stoi))


if __name__ == '__main__':
    evaluate('result_1.txt')
