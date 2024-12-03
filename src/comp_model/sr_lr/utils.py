#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:48:06 2021

@author: yl254115
"""
import pickle
import torch
from tqdm import tqdm
import numpy as np


def get_opposite_verb_form(verb):

    if verb.endswith('s'):  # singular
        if verb.endswith(('ses', 'shes', 'ches', 'x', 'z')):
            opposite_verb = verb[:-2]
        else:
            opposite_verb = verb[:-1]
    else:  # plural
        if verb.endswith(('s', 'sh', 'ch', 'x', 'z')):
            opposite_verb = verb + 'es'
        else:
            opposite_verb = verb + 's'
    return opposite_verb


def get_output_activations(model, path2activations, LR_units, SR_units):
    other_units = list(set(range(650, 1300))-set(LR_units) - set(SR_units))
    LR_units, SR_units, other_units = \
        np.asarray(LR_units), np.asarray(SR_units), np.asarray(other_units)

    embeddings_out = model.decoder.weight.data.cpu().numpy()
    bias_out = model.decoder.bias.data.cpu().numpy()
    model_activations = pickle.load(open(path2activations, 'rb'))
    model_activations = np.asarray(model_activations['hidden'])

    output_activations = {}
    output_activations['SR'] = np.dot(embeddings_out[:, SR_units-650],
                                      model_activations[:, SR_units, :])
    output_activations['LR'] = np.dot(embeddings_out[:, LR_units-650],
                                      model_activations[:, LR_units, :])

    return output_activations


def get_sr_lr_predictions(output_activations, df_meta, vocab):
    lexicon_size, num_trials, num_timepoints = output_activations['SR'].shape
    contribution_to_prediction = {}
    contribution_to_prediction['SR'] = np.zeros((num_trials, num_timepoints))
    contribution_to_prediction['LR'] = np.zeros((num_trials, num_timepoints))
    for i_trial, row in df_meta.iterrows():
        verb = row['v1']
        if row['violIndex']:
            verb_correct = get_opposite_verb_form(verb)
            verb_wrong = verb
        else:
            verb_correct = verb
            verb_wrong = get_opposite_verb_form(verb)

        if verb_correct not in vocab.word2idx \
                or verb_wrong not in vocab.word2idx:
            print(f'Skipping: {verb_correct}, {verb_wrong}')
            contribution_to_prediction['SR'][i_trial, :] = np.nan
            contribution_to_prediction['LR'][i_trial, ] = np.nan
            continue
    
        a_correct_verb_from_SR = output_activations['SR'][vocab.word2idx[verb_correct], i_trial, :]
        a_correct_verb_from_LR = output_activations['LR'][vocab.word2idx[verb_correct], i_trial, :]
    
        a_wrong_verb_from_SR = output_activations['SR'][vocab.word2idx[verb_wrong], i_trial, :]
        a_wrong_verb_from_LR = output_activations['LR'][vocab.word2idx[verb_wrong], i_trial, :]
    
        contribution_to_prediction['SR'][i_trial, :] = \
            np.log10(np.exp(1) ** (a_correct_verb_from_SR - a_wrong_verb_from_SR))
        contribution_to_prediction['LR'][i_trial, :] = \
            np.log10(np.exp(1) ** (a_correct_verb_from_LR - a_wrong_verb_from_LR))
    
    return contribution_to_prediction
