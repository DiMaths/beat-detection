#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detects onsets, beats and tempo in WAV files.

For usage information, call with --help.

Author of the skeleton: Jan Schlüter
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

from central_avg_envlope import moving_central_average
from spectral_diff import spectral_diff
from util import sliding_max, sliding_min, relative_spikes


def opts_parser():
    usage = \
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
    parser.add_argument('--plot_lvl',
                        type=int,
                        default=0,
                        help='Debugging plots during the run, higher lvl -> more extra plots')
    parser.add_argument('--method',
                        type=str,
                        default="central_avg_envelope",
                        help="Specifies the method to uses")
    parser.add_argument('--avg_window_size',
                        type=int,
                        default=50,
                        help="Relevant only for method='central_avg_envelope', size of sliding window")
    parser.add_argument('--gauss_smooth',
                        action='store_true',
                        help="Relevant only for method='central_avg_envelope', if given, apply not simple avg, "
                             "but weighted")
    parser.add_argument('--p_norm',
                        type=int,
                        default=2,
                        help="Relevant only for method='spectral_diff', L_p norm is applied for difference")
    parser.add_argument('--positive_only',
                        action='store_true',
                        help="Relevant only for method='spectral_diff', if given, only enlarging difference is taken "
                             "into account")
    parser.add_argument('--sliding_max_window_size',
                        type=int,
                        default=150,
                        help="Used for selecting odf peaks")
    parser.add_argument('--min_rel_jump',
                        type=float,
                        default=0.25,
                        help="Used for selecting odf peaks, local max is a peak only if local max / local min > 1 + min_rel_jump")
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
    if options.plot_lvl > 0:
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


def onset_detection_function(sample_rate, signal, fps, spect, magspect,
                             melspect, options):
    """
    Compute an onset detection function. Ideally, this would have peaks
    where the onsets are. Returns the function values and its sample/frame
    rate in values per second as a tuple: (values, values_per_second)
    """
    subsampling_factor = 1
    if options.method == "central_avg_envelope":
        subsampling_factor = options.avg_window_size
        values = moving_central_average(np.abs(signal),
                                        options.avg_window_size,
                                        gaussian_smoothing=options.gauss_smooth)[::subsampling_factor]
    
    elif options.method == "spectral_diff":
        subsampling_factor = sample_rate // fps 
        values = spectral_diff(spect, options.p_norm, options.positive_only)

    values_per_second = sample_rate / subsampling_factor
    return values, values_per_second


def detect_onsets(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """

    spikes = relative_spikes(odf, 
                             options.sliding_max_window_size, 
                             options.min_rel_jump, options.plot_lvl > 2)

    if options.plot_lvl > 1:
        plt.plot(sliding_max(odf, options.sliding_max_window_size), label="sliding max")
        plt.plot(sliding_min(odf, options.sliding_max_window_size), label="sliding min")
        
        plt.plot(odf, label="onset detection function")
        plt.plot(spikes, odf[spikes], 'o', label="onsets")
        all_local_maximas = relative_spikes(odf, options.sliding_max_window_size, 0.0)
        plt.plot(all_local_maximas, odf[all_local_maximas], 'x', label="local_maximas")
        plt.legend()
        plt.show()

    return spikes / odf_rate


def detect_tempo(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, options):
    """
    Detect tempo using any of the input representations.
    Returns one tempo or two tempo estimations.
    """
    differences = np.diff(onsets)

    if options.plot_lvl > 2:
            plt.hist(differences)
            plt.show()
    estimates = []
    for p in range(-3, 3):
        for base in [2,3,5,7]:
            range_min = base**p
            if range_min > 4:
                continue
            diffs_in_range = differences[(range_min <= differences) & (differences <= range_min * base)]
            if len(diffs_in_range) > len(differences)/10:
                est = np.median(diffs_in_range)
                while est > 0.75:
                    est /= 2
                while est < 0.25:
                    est *= 2
                estimates.append(est)
    
    tempo = 60 / np.median(estimates)


    return [tempo / 2, tempo]


def detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, tempo, options):
    """
    Detect beats using any of the input representations.
    Returns the positions of all beats in seconds.
    """

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
