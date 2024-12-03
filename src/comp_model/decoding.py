#!/usr/bin/env python
import os
import pickle
import argparse
import pandas as pd
from decoding.viz import plot_GAT
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from mne.decoding import (GeneralizingEstimator, cross_val_multiscore)


parser = argparse.ArgumentParser(description='Decode violation/congruence/etc')
# MODEL
parser.add_argument('--model', type=str,
                    default='../../models/hidden650_batch128_dropout0.2_lr20.0.pt')
parser.add_argument('--var-type', type=str, default='hidden',
                    choices=['hidden', 'cell'])
# STIMULI
parser.add_argument('--stimuli', default='../../data/stimuli/stimuli.csv',
                    help='Input sentences')
# CLASSIFICATION
parser.add_argument('--queries', default=None, nargs='*',
                    help='List of two queries for the two classes.')
args = parser.parse_args()

########
# Load #
########
print('Loading metadata')
df_meta = pd.read_csv(args.stimuli)
print('Loading activations...')
activations = '../../data/activations/%s_%s.pkl' % \
    (os.path.basename(args.model),'stimuli')
LSTM_activations = pickle.load(open(activations, 'rb'))
LSTM_activations = LSTM_activations[args.var_type]

num_trials, num_timepoints, num_units = len(LSTM_activations),\
                                        LSTM_activations[0][0].size,\
                                        len(LSTM_activations[0])
print(num_trials, num_timepoints, num_units)
# PREPARE DATA FOR CLASSIFICATION (X, y)
X = []
for i_trial, trial_data in enumerate(LSTM_activations):
    #print(trial_data.shape)
    trial_vector_t = []
    for i_t, t_data in enumerate(trial_data.T):
        trial_vector_t.append(t_data.reshape(-1, 1))  # flatten layers to a vec
    X.append(np.hstack(trial_vector_t))
X = np.asarray(X)

# STANDARDIZE
for u in range(X.shape[1]):
    scaler = StandardScaler()
    X[:, u, :] = scaler.fit_transform(X[:, u, :])

# QUERY
Xs, ys = [], []
for i_query, query in enumerate(args.queries):
    IXs = df_meta.query(query).index.to_numpy()
    Xs.append(X[IXs, :, :])
    ys.append(i_query * np.ones(len(IXs)))
    example_sentence = \
        df_meta.query(query)['sentence'].iloc[0].rstrip('\n').rstrip('.')
    words = example_sentence.split(' ')

X = np.vstack(Xs)
y = np.hstack(ys)

##################
# TRAIN AND EVAL #
##################
clf = RidgeClassifier(class_weight='balanced')
time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=-1,
                                 verbose=True)
scores = cross_val_multiscore(time_gen, X, y, cv=5, n_jobs=-1)
scores = np.mean(scores, axis=0)


########
# PLOT #
########
diag_scores = np.diagonal(scores)
title = ''
fig, axs = plot_GAT(scores, 6, words, title)
queries = '_vs_'.join(args.queries)
fname = f'GAT_{args.var_type}_{os.path.basename(args.model)}_{queries}.png'
fname = fname.replace(' ', '_')
plt.savefig(f'../../figures/gat/{fname}')
print(f'Saved to: ../../figures/gat/{fname}')
