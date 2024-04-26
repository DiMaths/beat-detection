#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates onset, beat and/or tempo predictions against ground truth.

For usage information, call with --help.

Author: Jan Schlüter
"""

import sys
import os
from argparse import ArgumentParser
from pathlib import Path
import json
from collections import defaultdict

import numpy as np
import mir_eval


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


def eval_onsets(truth, preds):
    """
    Computes the average onset detection F-score.
    """
    return sum(mir_eval.onset.f_measure(np.asarray(truth[k]['onsets']),
                                        np.asarray(preds[k]['onsets']),
                                        0.05)[0]
               for k in truth if k in preds) / len(truth)


def eval_tempo(truth, preds):
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

    return sum(mir_eval.tempo.detection(*prepare_truth(truth[k]['tempo']),
                                        prepare_preds(preds[k]['tempo']),
                                        0.08)[0]
               for k in truth if k in preds) / len(truth)


def eval_beats(truth, preds):
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
        print('Onsets F-score: %.4f' % eval_onsets(truth, preds))
    except KeyError:
        print('Onsets seemingly not included.')
    try:
        print('Tempo p-score: %.4f' % eval_tempo(truth, preds))
    except KeyError:
        print('Tempo seemingly not included.')
    try:
        print('Beats F-score: %.4f' % eval_beats(truth, preds))
    except KeyError:
        print('Beats seemingly not included.')


if __name__ == "__main__":
    main()

