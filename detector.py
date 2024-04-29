#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detects onsets, beats and tempo in WAV files.

For usage information, call with --help.

Author: Jan Schlüter
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
from scipy.io import wavfile
import librosa
import tqdm

import matplotlib.pyplot as plt

def opts_parser():
    usage =\
"""
Detects onsets, beats and tempo in WAV files.
"""
    parser = ArgumentParser(description=usage)
    parser.add_argument('indir',
            type=str,
            help='Directory of WAV files to process.')
    parser.add_argument('outfile',
            type=str,
            help='Output JSON file to write.')
    parser.add_argument('--plot',
            action='store_true',
            help='If given, plot something for every file processed.')
    return parser


def detect_everything(filename, options):
    """
    Computes some shared features and calls the onset, tempo and beat detectors.
    """
    # read wave file (this is faster than librosa.load)
    sample_rate, signal = wavfile.read(filename)

    # convert from integer to float
    if signal.dtype.kind == 'i':
        signal = signal / np.iinfo(signal.dtype).max

    # convert from stereo to mono (just in case)
    if signal.ndim == 2:
        signal = signal.mean(axis=-1)

    # compute spectrogram with given number of frames per second
    fps = 70
    hop_length = sample_rate // fps
    spect = librosa.stft(
            signal, n_fft=2048, hop_length=hop_length, window='hann')

    # only keep the magnitude
    magspect = np.abs(spect)

    # compute a mel spectrogram
    melspect = librosa.feature.melspectrogram(
            S=magspect, sr=sample_rate, n_mels=80, fmin=27.5, fmax=8000)

    # compress magnitudes logarithmically
    melspect = np.log1p(100 * melspect) 

    # compute onset detection function
    odf, odf_rate = onset_detection_function(
            sample_rate, signal, fps, spect, magspect, melspect, options)

    # detect onsets from the onset detection function
    onsets = detect_onsets(odf_rate, odf, options)

    # detect tempo from everything we have
    tempo = detect_tempo(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, options)

    # detect beats from everything we have (including the tempo)
    beats = detect_beats(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, tempo, options)

    # plot some things for easier debugging, if asked for it
    if options.plot:
        fig, axes = plt.subplots(3, sharex=True)
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(filename)
        axes[0].set_title('melspect')
        axes[0].imshow(melspect, origin='lower', aspect='auto',
                       extent=(0, melspect.shape[1] / fps,
                               -0.5, melspect.shape[0] - 0.5))
        axes[1].set_title('onsets')
        axes[1].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in onsets:
            axes[1].axvline(position, color='tab:orange')
        axes[2].set_title('beats (tempo: %r)' % list(np.round(tempo, 2)))
        axes[2].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in beats:
            axes[2].axvline(position, color='tab:red')
        plt.show()

    return {'onsets': list(np.round(onsets, 3)),
            'beats': list(np.round(beats, 3)),
            'tempo': list(np.round(tempo, 2))}


def moving_central_average(x: np.ndarray, w: int, mode: str = 'same', gaussian_smoothing: bool = False) -> np.ndarray:
    """
    Computes moving average of sequence x with window size w
    @param x: np.ndarray
    @param w: int - window size
    @param mode: str 
        - if 'same' then returns same length array by adding zeros to ends
        - if 'valid' return array of length (len(x) - w + 1)
    @param gaussian_smoothing: bool 
        - if True weights are taken from normal distribution pdf
    """
    if mode not in ["same", "valid"]:
        raise ValueError("Wrong mode of moving central averaging, allowed are 'same' and 'valid'.")
    if w <= 0:
        raise ValueError(f"Window size for moving central average must be positive integer, but got {w}.") 
    
    weights = np.ones(w)
    if gaussian_smoothing:
        eq_points = np.linspace(0, 1, num=w+2)[1:]
        weights = [(2*np.pi)**(-0.5) * np.exp(-0.5 * point**2) for point in eq_points]
        max_weight = max(weights)
        weights = [w/max_weight for w in weights]

    return np.convolve(x, weights, mode) / sum(weights)

def sliding_max(x: np.ndarray, w: int) -> np.ndarray:
    """
    Computes sliding max (max in sliding window)
    @param x: np.ndarray - original sequence
    @param w: int - size of the sliding window

    @returns y: np.ndarray - array of the same size as x
    """
    if w <= 0:
        raise ValueError(f"Window size must be positive integer, but got {w}.")

    # y is copy of x plus padded 0s
    y = np.zeros(x.shape[0] + w - 1)
    w_half = int(w/2)
    y[w_half:-(w-w_half-1)] = x  

    # recursive implementation allows O(len(x) * log(w)) complexity 
    # instead of O(len(x) * w)
    current_w = 1
    while current_w < w:
        step = min(current_w, w-current_w)
        y = np.maximum(y[step:], y[:-step])
        current_w += step
    return y

def onset_detection_function(sample_rate, signal, fps, spect, magspect,
                             melspect, options):
    """
    Compute an onset detection function. Ideally, this would have peaks
    where the onsets are. Returns the function values and its sample/frame
    rate in values per second as a tuple: (values, values_per_second)
    """

    values = moving_central_average(np.abs(signal), 50, gaussian_smoothing=True)[::50]
    """
    plt.plot(np.abs(signal), label="original magnitude")
    plt.plot(np.arange(0,len(signal),50), values, label="moving average")
    plt.legend()
    plt.show()
    """
    values_per_second = sample_rate / 50
    return values, values_per_second


def detect_onsets(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    local_maximas = np.where(sliding_max(odf, 150) == odf)[0]
    
    """plt.plot(sliding_max(odf, 5_000), label="sliding max")
    plt.plot(odf, label="onset detection function")
    plt.legend()
    plt.show()"""
    
    return local_maximas / odf_rate


def detect_tempo(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, options):
    """
    Detect tempo using any of the input representations.
    Returns one tempo or two tempo estimations.
    """    
    # tempo = 60 / (onsets[1] - onsets[0]) # old dummy placeholder
    # plt.hist(np.diff(onsets))
    # plt.show()
    
    differences = np.diff(onsets)
    tempo = 60 / differences[(0.25 <= differences) & (differences<=0.5)].mean()
    
    return [tempo / 2, tempo]


def detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, tempo, options):
    """
    Detect beats using any of the input representations.
    Returns the positions of all beats in seconds.
    """
    """current = onsets[1]
    beats = [current]
    while current <= onsets[-1]:
        current += tempo[0]/60
        beats.append(current)"""
    beats = onsets
    return beats


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()

    # iterate over input directory
    indir = Path(options.indir)
    infiles = list(indir.glob('*.wav'))
    if tqdm is not None:
        infiles = tqdm.tqdm(infiles, desc='File')
    results = {}
    for filename in infiles:
        results[filename.stem] = detect_everything(filename, options)

    # write output file
    with open(options.outfile, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

