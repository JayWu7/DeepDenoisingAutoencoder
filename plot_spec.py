#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELEC-E5500 Speech Processing -- Autumn 2020 Python Exercise 1:
Basics of speech processing and analysis in Python.

Recommended to use a virtual environment to have a clear management of the libraries used in the exercises.

Python version: 3.5 or higher

To make sure all the packages are up-to-date for the exercise, run the script Update_Packages_ex1.py.
"""
import os.path as path
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import ex1_windowing as win
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os

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

    if not (Fs == Fs_target):
        in_sig = sig.resample_poly(in_sig, Fs_target, Fs)
        Fs = Fs_target

    ## 1.3. Split the data sequence into windows.
    # Implement windowing function in ex1_windowing_solution.py
    frame_length = np.int(np.around(0.025 * Fs))  # 25ms in samples
    hop_size = np.int(np.around(0.0125 * Fs))  # 12.5 ms in samples (50% overlap)
    window_types = ('rect', 'hann', 'cosine', 'hamming')
    frame_matrix = win.ex1_windowing(in_sig, frame_length, hop_size, window_types[3])  # Windowing

    return frame_matrix

def calculate_spec(frame_matrix):
    frame_matrix_fft = np.fft.rfft(frame_matrix, axis=0, n=Nfft)
    frame_matrix_fft = 20 * np.log10(np.absolute(np.flipud(frame_matrix_fft)))
    return frame_matrix_fft

def plot(result_dir):
    reg_root = path.join(data_root, 'REG/')
    source_root = path.join(data_root, 'Source/')
    target_root = path.join(data_root, 'Target/')
    f_axis = np.divide(range(np.int(Nfft / 2) + 1), (Nfft / 2) / (Fs_target / 2))
    audios = os.listdir(reg_root)
    for audio in audios:
        reg_audio = path.join(reg_root, audio)
        source_audio = path.join(source_root, audio)
        target_audio = path.join(target_root, audio)

        reg_matrix = get_matrix(reg_audio)
        source_matrix = get_matrix(source_audio)
        target_matrix = get_matrix(target_audio)

        reg_spec = calculate_spec(reg_matrix)
        source_spec = calculate_spec(source_matrix)
        target_spec = calculate_spec(target_matrix)

        plt.figure(1, figsize=(16, 12))
        plt.subplot(3, 1, 1)
        plt.imshow(reg_spec, aspect='auto')
        ytickpos = np.flipud([0, np.int(Nfft / 8), np.int(Nfft / 4), np.int(Nfft * 3 / 8), np.int(Nfft / 2)])
        plt.yticks([0, Nfft / 8, Nfft / 4, Nfft * 3 / 8, Nfft / 2], f_axis[ytickpos])
        plt.title('Enhanced Spectrogram')
        plt.subplot(3, 1, 2)
        plt.imshow(source_spec, aspect='auto')
        ytickpos = np.flipud([0, np.int(Nfft / 8), np.int(Nfft / 4), np.int(Nfft * 3 / 8), np.int(Nfft / 2)])
        plt.yticks([0, Nfft / 8, Nfft / 4, Nfft * 3 / 8, Nfft / 2], f_axis[ytickpos])
        plt.title('Noisy Spectrogram')
        plt.ylabel('Frequency (Hz)')
        plt.subplot(3, 1, 3)
        plt.imshow(target_spec, aspect='auto')
        ytickpos = np.flipud([0, np.int(Nfft / 8), np.int(Nfft / 4), np.int(Nfft * 3 / 8), np.int(Nfft / 2)])
        plt.yticks([0, Nfft / 8, Nfft / 4, Nfft * 3 / 8, Nfft / 2], f_axis[ytickpos])
        plt.title('Clean Spectrogram')
        plt.xlabel('Frame number')

        save_path = path.join(result_dir, audio[:-4])
        plt.savefig('{}.png'.format(save_path))


if __name__ == '__main__':
    plot('./result_1')


