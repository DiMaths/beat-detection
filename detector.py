#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detects onsets, beats and tempo in WAV files.

For usage information, call with --help.

Author of the skeleton: Jan Schlüter
Authors: Dmytro Borysenkov
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

from evaluate import read_data

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
    parser.add_argument('--training',
                        action='store_true',
                        help="If plot lvl > 0, shows ground truth beats, onsets and tempo on the plots")
    parser.add_argument('--method',
                        type=str,
                        default="spectral_diff",
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
                        default=1,
                        help="Relevant only for method='spectral_diff', L_p norm is applied for difference")
    parser.add_argument('--positive_only',
                        action='store_true',
                        help="Relevant only for method='spectral_diff', if given, only enlarging difference is taken "
                             "into account")
    parser.add_argument('--sliding_max_window_size',
                        type=int,
                        default=12,
                        help="Used for selecting odf peaks")
    parser.add_argument('--min_rel_jump',
                        type=float,
                        default=0.0,
                        help="Used for selecting odf peaks, local max is a peak only if local max / local min > 1 + min_rel_jump")
    return parser


def detect_everything(filename, options,model_CNN):
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
    fps = 100  # should be 70, I just temporaly replace it to be 100
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

    # Dmytro, could you please set that this code will work if it asked so in parser, cause
    # I never worked with it
    onsets = detect_onsets_with_CNN(filename,model_CNN)
    # print("-------------------------------")
    # print(onsets)
    # print("-------------------------------")

    # onsets_2 = detect_onsets(odf_rate, odf, options)
    # print("-------------------------------")
    # print(onsets_2)
    # print("-------------------------------")

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
        fig, axes = plt.subplots(3, sharex=True, figsize=(12,7))
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(filename)
        axes[0].set_title('melspect')
        axes[0].imshow(melspect, origin='lower', aspect='auto',
                       extent=(0, melspect.shape[1] / fps,
                               -0.5, melspect.shape[0] - 0.5))
        
        axes[1].set_title('onsets')
        axes[1].plot(np.arange(len(odf)) / odf_rate, odf)
    
        for position in onsets:
            axes[1].axvline(position, color='tab:red')
        if options.training:
            for position in GT[filename.stem]['onsets']:
                axes[1].axvline(position, color='tab:green', linestyle='--')
        title = 'beats (tempo: %r)' % list(np.round(tempo, 2))
        if options.training:
            title += ' (ground truth tempo: %r)'  % list(np.round(GT[filename.stem]['tempo'], 2))
        axes[2].set_title(title)
        axes[2].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in beats:
            axes[2].axvline(position, color='tab:red')
        if options.training:
            for position in GT[filename.stem]['beats']:
                axes[2].axvline(position, color='tab:green', linestyle='--')
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
    elif options.method == "melspect_diff":
        subsampling_factor = sample_rate // fps
        values = spectral_diff(melspect, options.p_norm, options.positive_only, melscaled=True)
    values_per_second = sample_rate / subsampling_factor
    return values, values_per_second


def detect_onsets(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """

    spikes = relative_spikes(odf, 
                             options.sliding_max_window_size, 
                             options.min_rel_jump, options.plot_lvl > 9)


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
            plt.suptitle("Onset differences distribution")
            plt.show()
    estimates = []
    for p in range(-3, 3):
        for base in [2,3,5,7]:
            range_min = base**p
            if range_min > 4 or range_min < 0.03:
                continue
            diffs_in_range = differences[(range_min <= differences) & (differences <= range_min * base)]
            if len(diffs_in_range) > len(differences)/20:
                est = np.median(diffs_in_range)
                while est > 0.75: # until tempo estimate is < 80 (tempo/2 < 40)
                    est /= 2
                while est < 0.25: # until tempo estimate is > 240 (tempo/2 > 120)
                    est *= 2
                estimates.append(est)
    
    if len(estimates) > 0:
        tempo = 60 / np.median(estimates)
    else:
        if options.plot_lvl > 0:
            plt.hist(differences)
            plt.title("Not enough onsets detected to estimate the tempo, tempo =  [60, 120] returned as default")
            plt.suptitle("Onset differences distribution")
            plt.show()
        else:
            print("Not enough onsets detected to estimate the tempo, tempo =  [60, 120] returned as default")
            
        return [60., 120.]    
    return [tempo / 2, tempo]


def detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, tempo, options):
    """
    Detect beats using any of the input representations.
    Returns the positions of all beats in seconds.
    """
    beats = onsets
    return beats


def detect_onsets_with_CNN(test,model_CNN,threshold=0.63):
    dataset_test = CNN_dataset([test],test_tr = 1)
    train_dataloader = DataLoader(dataset_test, batch_size=128, shuffle=True)
    testing_np = []
    for input in train_dataloader:
        input = input.to('cuda')
        output = model_CNN(input,istraining=False)
        output = output.detach().cpu().numpy()
        testing_np.append(output)

    testing_np = np.concatenate(testing_np,axis=0)
    testing_np = testing_np.reshape(-1)
    peaks, _ = find_peaks(testing_np,distance=4)
    onsets = []

    for peak in peaks:
        if testing_np[peak] > threshold:
            onsets.append(peak)
    
    onsets = np.array(onsets)/100
    return onsets
    


import torch
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
from tqdm import tqdm
import sklearn
import torch.nn as nn
from sklearn.metrics import accuracy_score
from model_CNN import CNN_Model
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

class CNN_dataset(torch.utils.data.Dataset):

    def __init__(self,infiles,test_tr = 0):
        self.test_train = test_tr
        data_df = [np.zeros((3,80,7),dtype=float)] # padding of 7
        if self.test_train == 0:
            labels_df = []

        for filename in infiles:
            if self.test_train == 0:
                data, label = self.get_np_data(filename)
                labels_df.append(label)
            else:
                data = self.get_np_data(filename)
            data_df.append(data)

        data_df.append(np.zeros((3,80,7),dtype=float)) # padding of 7

        self.data_df = np.concatenate(data_df,axis=2)
        if self.test_train == 0: 
            self.labels_df = np.concatenate(labels_df,axis=0).reshape(-1,1)
            print(self.labels_df.shape)


    def __len__(self):
        return self.data_df.shape[2]-14

    def __getitem__(self, idx: int):
        if self.test_train == 0:
            label = torch.Tensor(self.labels_df[idx])
            item = torch.Tensor(self.data_df[:,:,idx:idx+15])
            return item, label
        else:
            item = torch.Tensor(self.data_df[:,:,idx:idx+15])
            return item
    
    def get_np_data(self,filename):
        sample_rate, signal = wavfile.read(filename)

        # convert from integer to float
        if signal.dtype.kind == 'i':
            signal = signal / np.iinfo(signal.dtype).max

        # convert from stereo to mono (just in case)
        if signal.ndim == 2:
            signal = signal.mean(axis=-1)

        fps_ = 100
        hop_length_ = sample_rate // fps_

        base_name_with_extension = os.path.basename(filename)
        base_name = os.path.splitext(base_name_with_extension)[0]

        melspects_3 = []
        
        n_ffts = [1024,2048,4096]
        for n_fft in n_ffts:

            spect_ = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length_, window='hann')

            # Only keep the magnitude
            magspect_ = np.abs(spect_)

            # compute a mel spectrogram
            melspect_ = librosa.feature.melspectrogram(S=magspect_, sr=sample_rate, n_mels=80, fmin=27.5, fmax=16000)
            
            # compress magnitudes logarithmically
            melspect_ = np.log1p(100 * melspect_)
            melspects_3.append(melspect_)


        melspects_3 = np.array(melspects_3)

        if self.test_train == 0:
            # ground truth ------------- #  ----------- # ------------ #
            onsets_ground_t = np.around((np.array(GT[base_name]["onsets"])*100)).astype(int)

            onsets_ground_t_3 = []
            for i in onsets_ground_t:
                onsets_ground_t_3.append(i-1)
                onsets_ground_t_3.append(i)
                onsets_ground_t_3.append(i+1)

            # basic deletion of 1st and last element
            if onsets_ground_t_3[0] < 1:
                onsets_ground_t_3 = onsets_ground_t_3[1:]
            if onsets_ground_t_3[-1] > melspects_3.shape[2]-1:
                onsets_ground_t_3 = onsets_ground_t_3[:-1]

            ground_truth_onsets = np.zeros(melspects_3.shape[2])
            ground_truth_onsets[onsets_ground_t_3] = 1
            # ground truth ------------- #  ----------- # ------------ #

            # plt.imshow(melspects_3[0], aspect='auto', cmap='viridis')
            # plt.show()

            return melspects_3, ground_truth_onsets
        
        return melspects_3



def training_step(network, optimizer, data, targets, loss_fn):
    optimizer.zero_grad()
    output = network(data,istraining=True)
    labels_processed = targets.flatten().float().reshape(-1,1)
    # print(labels_processed.shape,output.shape)
    loss = loss_fn(output, labels_processed)
    loss.backward()
    optimizer.step()
    return loss.item()

    
def get_metric(network, test_dataloader,loss_fn,threshold = 0.63):
    network.eval().to('cuda')
    running_loss = 0.
    accuracy = 0.
    counter = 0
    f1_scores = 0.
    for i, data in tqdm(enumerate(test_dataloader)):
        input, true_labels = data
        input = input.to('cuda')
        true_labels = true_labels.to('cuda')
        output = network(input,istraining=False)
        labels_processed = true_labels.flatten().float().reshape(-1,1)
        loss = loss_fn(output, labels_processed)
        running_loss += loss.item()

        # for now I will impement it without hamming window as I am supposed
        # but only with threshold
        output = output.detach().cpu().numpy()
        output = (output>threshold).astype(int)
        # output, _ = find_peaks(output,distance=4)
 
        labels_processed = labels_processed.detach().cpu().numpy()
        # labels_processed, _ = find_peaks(labels_processed,distance=4)
        acc = accuracy_score(labels_processed, output)
        f1 = sklearn.metrics.f1_score(labels_processed, output,average="weighted")
        f1_scores += f1
        accuracy += acc
        counter += 1
    print('Loss =', running_loss / counter)
    print('Accuracy = ', 100 * (accuracy / counter))
    print('F1_score = ', 100 * (f1_scores / counter))


def training_loop(
        network: torch.nn.Module,
        train_dataloader: torch.utils.data.dataloader.DataLoader,
        test_dataloader: torch.utils.data.dataloader.DataLoader,
        num_epochs: int,
        show_progress: bool = True) -> tuple[list, list]:
    
    loss_fn = nn.MSELoss()
    device = "cuda"
    device = torch.device(device)
    print(torch.__version__)
    print(torch.cuda.get_device_name(0))

    if not torch.cuda.is_available():
        print("CUDA IS NOT AVAILABLE")
        device = torch.device("cpu")
    losses = []

    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0002)

    for epoch in tqdm(range(num_epochs), desc="Epoch", position=0, disable= (not show_progress)):

        network.train().to('cuda')
        running_loss = 0.
        last_loss = 0.
        for i, data in tqdm(enumerate(train_dataloader), desc="Minibatch", position=1, leave=False, disable= (not show_progress)):
            inputs, targets = data
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            loss = training_step(network, optimizer, inputs, targets,loss_fn)
            running_loss += loss
            if i % 1000 == 999:
                last_loss = running_loss / 999 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        losses.append(last_loss)
        print('\n')
        get_metric(network, test_dataloader,loss_fn)
        print('\n\n\n')
            
    return network, losses

def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()

    # iterate over input directory
    indir = Path(options.indir)
    global GT 
    if options.training:
        GT = read_data(indir)
    infiles = list(indir.glob('*.wav'))

    # Train_ data -------------- # -------------- # -------------- # - # - # - # - # - #
    

    indir_train_c = Path("train_extra/")
    infiles_train_c = list(indir_train_c.glob('*.wav'))
    print(len(GT.keys()),"GT_befor")
    if options.training:
        dict_extra = read_data(indir_train_c)
        GT.update(dict_extra)
    print(len(GT.keys()),"GT")
    print(len(infiles_train_c),"indir_train_c")
    # CNN -------------- # -------------- # -------------- # - # - # - # - # - #

    print("Training CNN")
    # infiles_train_c = infiles_train_c[:30]
    train, test = train_test_split(infiles_train_c, test_size=0.02)
    # python detector.py train/ output.json --plot_lvl 0 --training
    dataset_train = CNN_dataset(train)
    print("Train data is ready")
    dataset_val = CNN_dataset(test)
    print("Val data is ready")

    print("LEN of data:", len(dataset_train))
    print("LEN of data:", len(dataset_val))
    print("FISRT ELEMENT", dataset_train[0][0].shape)

    model = CNN_Model()

    train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=False)
    
    model,_ = training_loop(model,train_dataloader,test_dataloader,5,True) # network

    # CNN -------------- # -------------- # -------------- # - # - # - # - # - #

    if tqdm is not None:
        infiles = tqdm(infiles, desc='File')
    results = {}
    for filename in infiles:
        results[filename.stem] = detect_everything(filename, options,model)

    # write output file
    with open(options.outfile, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
