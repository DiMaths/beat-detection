#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates onset, beat and/or tempo predictions against ground truth.

For usage information, call with --help.

Author of the skeleton: Jan Schlüter
Author: Dmytro Borysenkov
"""

import sys
import os
from argparse import ArgumentParser
from pathlib import Path
import json
from collections import defaultdict

import numpy as np
import mir_eval
import matplotlib.pyplot as plt


def opts_parser():
    usage =\
"""Evaluates onset, beat and/or tempo predictions against ground truth.

"""
    parser = ArgumentParser(description=usage)
    parser.add_argument('groundtruth',
            type=str,
            help='The ground truth directory of .gt files or a JSON file.')
    parser.add_argument('predictions',
            type=str,
            help='The predictions directory of .pr files or a JSON file.')
    parser.add_argument('--extended',
            action='store_true',
            default=False,
            help="Enables detailed prints of evaluation")
    return parser


def read_data(path, extension='.gt'):
    """
    Read a directory or JSON file of onsets, beats and/or tempo.
    """
    path = Path(path)
    if path.is_file():
        with open(path, 'r') as f:
            return json.load(f)
    else:
        data = defaultdict(dict)
        for filename in path.glob('*%s' % extension):
            stem, kind, _ = filename.name.rsplit('.', 2)
            with open(filename, 'r') as f:
                if kind == 'tempo':
                    values = [float(value)
                              for value in f.read().rstrip().split()]
                else:
                    values = [float(line.rstrip().split()[0])
                              for line in f if line.rstrip()]
            data[stem][kind] = values
        return data


def eval_onsets(truth, preds, options):
    """
    Computes the average onset detection F-score.
    """
    onset_evals = dict()
    for k in truth:
        if k in preds:
            onset_evals[k] = mir_eval.onset.f_measure(
                                np.asarray(truth[k]['onsets']),
                                np.asarray(preds[k]['onsets']),
                                0.05)
    
    f_scores = [onset_evals[k][0] for k in onset_evals]

    if options.extended:
        bad_estimates = dict()
        f_score_avg = np.mean(f_scores)
        f_score_variance = np.var(f_scores)
        for k in onset_evals:
            if onset_evals[k][0] < f_score_avg - f_score_variance:
                bad_estimates[k] = onset_evals[k]

        bad_f_scores = [onset_evals[k][0] for k in bad_estimates]
        bad_precisions = [onset_evals[k][1] for k in bad_estimates]
        bad_recalls = [onset_evals[k][2] for k in bad_estimates]

        bad_estimates_labels = [k if len(k)< 25 else f"{k[:10]}...{k[-10:]}" for k in list(bad_estimates.keys())]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 7), sharex=True, sharey=True)
        ax.bar(bad_estimates_labels, bad_f_scores, color="orange", label='F-score')
        ax.set_ylabel("f_score")
        
        ax.plot(bad_estimates_labels, bad_precisions, color="red", marker='x', linestyle='', label='Precision')
        ax.set_ylabel("precision")

        ax.plot(bad_estimates_labels, bad_recalls, color="blue", marker='o', linestyle='', label='Recall')
        ax.set_ylabel("recall")

        ax.set_xlim(-1, len(bad_estimates) + 1)
        ax.set_ylim(0, 1.15)
        ax.axhline(1, -1, len(bad_estimates_labels) + 2, color='black', linestyle='--')
        ax.set_xticks(range(len(bad_estimates_labels)))
        ax.set_xticklabels(bad_estimates_labels, rotation=90, ha='center')
        plt.legend(loc='upper center', ncol=3, prop={'size': 14})
        fig.suptitle(f"Onsets Estimation Extended Eval Plot of the worst onsets f-score  samples (total: {len(bad_estimates)})")
        fig.tight_layout()
        plt.show()

        print(f"Poorly-guessed onsets of: {[k for k in bad_estimates]}")
        print()

    return sum(f_scores) / len(truth)


def eval_tempo(truth, preds, options):
    """
    Computes the average tempo estimation p-score.
    """
    def prepare_truth(tempi):
        if len(tempi) == 3:
            tempi, weight = tempi[:2], tempi[2]
        else:
            tempi, weight = [tempi[0] / 2., tempi[1]], 0.
        return np.asarray(tempi), weight

    def prepare_preds(tempi):
        if len(tempi) < 2:
            tempi = [tempi[0] / 2., tempi[0]]
        return np.asarray(tempi)
    
    prepared_preds = dict()
    prepared_truth = dict()
    prepared_weights = dict()
    
    tempo_evals = dict()
    for k in truth:
        if k in preds:
            prepared_preds[k] = prepare_preds(preds[k]['tempo'])
            prepared_truth[k], prepared_weights[k] = prepare_truth(truth[k]['tempo'])
            tempo_evals[k] = mir_eval.tempo.detection(
                                    prepared_truth[k], 
                                    prepared_weights[k],
                                    prepared_preds[k],
                                    0.08)
    p_scores = [tempo_evals[k][0] for k in tempo_evals]

    if options.extended:
        bad_estimates = dict()
        worst_estimates = dict()
        p_score_avg = np.mean(p_scores)
        for k in tempo_evals:
            if tempo_evals[k][1] < p_score_avg and not tempo_evals[k][2]:
                    if tempo_evals[k][0] > 0:
                        bad_estimates[k] = tempo_evals[k]
                    else:
                        worst_estimates[k] = (prepared_truth[k][0] - prepared_preds[k][0]) / prepared_truth[k][0]
        
        bad_p_scores = [tempo_evals[k][0] for k in bad_estimates]
        
        bad_estimates_labels = [k if len(k)< 25 else f"{k[:10]}...{k[-10:]}" for k in list(bad_estimates.keys())]
        worst_estimates_labels = [k if len(k)< 25 else f"{k[:10]}...{k[-10:]}" for k in list(worst_estimates.keys())]
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 7))
        axs[0].bar(bad_estimates_labels, bad_p_scores, color="orange")
        axs[0].set_xticks(range(len(bad_estimates_labels)))
        axs[0].set_xticklabels(bad_estimates_labels, rotation=90, ha='center')
        axs[0].set_ylabel("p_score")
        axs[0].set_title(f"p-scores of 'half-guessed' (total: {len(bad_estimates)}) samples")
        axs[0].set_xlim(-1, len(bad_estimates) + 1)
        axs[0].set_ylim(0, 1)

        axs[1].bar(worst_estimates_labels, 
                   [abs(worst_estimates[k]) for k in worst_estimates], 
                   color=["red" if worst_estimates[k] > 0 else "blue" for k in worst_estimates])
        axs[1].set_xticks(range(len(worst_estimates_labels)))
        axs[1].set_xticklabels(worst_estimates_labels, rotation=90, ha='center')
        axs[1].set_ylabel("(gt[0] - pr[0]) / gt[0]")
        axs[1].set_title(f"Relative errors of misspredicted (p_score = 0) samples (in total: {len(worst_estimates)}). Overestimated - red, under- blue.")
        axs[1].set_xlim(-1, len(worst_estimates) + 1)
        fig.suptitle("Tempo Estimation Extended Eval Plot")
        fig.tight_layout()
        plt.show()

        print(f"Half-guessed: {[k for k in bad_estimates]}")
        print()
        print(f"Misspredicted completely: {[k for k in worst_estimates]}")
        print()

    return sum(p_scores) / len(truth)


def eval_beats(truth, preds, options):
    """
    Computes the average beat detection F-score.
    """
    return sum(mir_eval.beat.f_measure(np.asarray(truth[k]['beats']),
                                       np.asarray(preds[k]['beats']),
                                       0.07)
               for k in truth if k in preds) / len(truth)


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()

    # read ground truth
    truth = read_data(options.groundtruth, extension='.gt')

    # read predictions
    preds = read_data(options.predictions, extension='.pr')

    # evaluate
    try:
        print(f"Onsets F-score: {eval_onsets(truth, preds, options): .4f}")
        print("-" * 20)
    except KeyError:
        print("Onsets seemingly not included.")
    try:
        print(f"Tempo p-score: {eval_tempo(truth, preds, options): .4f}")
        print("-" * 20)
    except KeyError:
        print("Tempo seemingly not included.")
    try:
        print(f"Beats F-score: {eval_beats(truth, preds, options): .4f}")
        print("-" * 20)
    except KeyError:
        print("Beats seemingly not included.")


if __name__ == "__main__":
    main()

