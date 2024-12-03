#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:13:53 2021

@author: yl254115
"""

import os
import argparse
import glob
import pandas as pd
from utils import data

parser = argparse.ArgumentParser('Merge subject stimulus files')
parser.add_argument('--vocabulary', default='../../models/vocab.txt')
parser.add_argument('--path2stimuli', default='../../data/stimuli')
parser.add_argument('--path2output', default='../../data/stimuli')
args = parser.parse_args()

# Create list of subjects from which to take stimuli
subject_list = [f'S{n:02}' for n in range(1, 11)]
# Load model vocab
vocab = data.Dictionary(args.vocabulary)

def all_in_vocab(row):
    return all(w in vocab.word2idx
               for w in row['sentence'].rstrip('\n').rstrip('.').split())


dfs = []
for subject_folder in subject_list:
    fns = glob.glob(os.path.join(args.path2stimuli,
                                 subject_folder,
                                 'curated', '*.csv'))
    assert len(fns) == 1
    df = pd.read_csv(fns[0])
    # Remove rows whose sentence has some words not in model's vocab
    df['all_in_vocab'] = df.apply(lambda row: all_in_vocab(row), axis=1)
    df = df[df['all_in_vocab'] > 0]
    dfs.append(df)
pd.concat(dfs).to_csv(os.path.join(args.path2output, 'stimuli.csv'),
                      index=False, header=True)
