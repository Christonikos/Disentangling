#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 03:25:30 2019

@author: czacharo
"""

# -----------------------------------------------------------------------------------#
# -------------------------#
# ---- IMPORT MODULES ----#
# -------------------------#
import os
import argparse
import pandas as pd
from lexicon import construct_lexicon # function to create the lexicon
from random_stim_generator import gen_sent # function to create the sentences
from tqdm import tqdm # add progress bar to the for pools
# -----------------------------------------------------------------------------------#

# --------------------------------------------------------#
# -------------------------#
# ---- PREFERENCES --------#
# -------------------------#
parser = argparse.ArgumentParser(description = 'Stimulus generator for English stimuli')
parser.add_argument('-v', '--verbose', type=bool, default=0, help='View the created dictionaries')
parser.add_argument('-p', '--root_path', type = str, default = os.path.join(
    os.sep,
    'home',
    'christos',
    'Projects',
    'language-local-global',
    ), help='The default rootpath')
parser.add_argument('-M', '--method', type=str, default='iEEG')

# Get the default arguments
args = parser.parse_args()
# --------------------------------------------------------#

# --------------------------------------------------------#
# Load the lexicon
words    = construct_lexicon(args.root_path, args.verbose)
# --------------------------------------------------------#







# generate N number of random sentences/condition 
n = 10
# shuffle these N sentences and from them select M:
m = 2

# Set Parameters ##############################
method                  = args.method
if method == 'MEG':
        n_blocks           = 10 # 
        n_trials           = m  # Get m trial(s) per condition for (K) blocks. 
else:
        n_blocks           = 5 
        n_trials           = m 

# Generate stimuli for the following number of subjects:
n_subjects                 =  20



stimuli_df = pd.DataFrame()
stimuli = {}
print('Generating stimuli per subject... ')
for subjID in tqdm(range(1,n_subjects+1)):
        if subjID <= 9:
                curr_subject = ('subj_0' + str(subjID))
        else:
                curr_subject = ('subj_' + str(subjID))
        stimuli[curr_subject] = {}
        for blockID in range(0,n_blocks):
            curr_block = ('b_' + str(blockID))
            stimuli[curr_subject][curr_block] = {} 
            new_pool = pd.DataFrame()
            cond_pool = pd.DataFrame()
            for i in range(0,2):
                for condition in ['GSLS','GSLD', 'GDLS','GDLD']:
                    for embedding in ['pp','objrc']:
                        for trial_type in ['syntactic','semantic']:
                            if (trial_type == 'semantic') and (embedding == 'objrc'):
                                continue
                            for gn in ['sing','plur']:
                                #----------@OBJECTIVE: MAXIMIZE ENTROPY: ----------------------------------#
                                # check if the the (M) selected sentences have different values for 
                                # the features n1,n2, v1, IF NOT, REPEAT THE PROCESS UNTIL THIS IS THE CASE:
                                while True:
                                    ### GENERATE SENTENCES PER CONDITION WITH RANDOM SELECTION FROM THE LEXICON ###
                                    generated_sentences = gen_sent(condition, words, embedding, trial_type, gn, 'random', n, 0)
                                    ### SHUFFLE THE (N) GENERATED SENTENCES PER CONDITION #######
                                    shuffled_sentences = generated_sentences.sample(frac=1)
                                    #### SELECT (M) WITHOUT REPLACEMENT ###############
                                    curr_trials = shuffled_sentences.sample(n = m,  replace = False)
                                    if trial_type == 'syntactic':
                                        # check the required objective - only break if true
                                        if (
                                               (curr_trials['n1'].nunique() == 2) \
                                           and (curr_trials['n2'].nunique() == 2)\
                                           and (curr_trials['v1'].nunique() == 2)
                                           ):
                                            break
                                    # -------- @SEMANTICS ONLY: WE ALSO WANT THE ANIMACY PAIRS TO BE DIFFERENT -------- #
                                    elif trial_type == 'semantic':
                                        # check the required objective - only break if true
                                        if (
                                               (curr_trials['n1'].nunique() == 2) \
                                           and (curr_trials['n2'].nunique() == 2)\
                                           and (curr_trials['v1'].nunique() == 2)\
                                           and (curr_trials['pair_index'].nunique() == 2)
                                           ):
                                            break
                                    
                                # shuffle the order of the conditions per mini-block to avoid order effects
                                cond_pool = cond_pool.append(curr_trials).sample(frac=1)# Append to the stimuli dictionary
            new_pool = new_pool.append(cond_pool)
            stimuli[curr_subject][curr_block] = new_pool
                    
                
            ###############################################
            ######## OUTPUT ###############################
            ###############################################
            # Per method/subject: 2 folders (Syntax & Semantics)
            # Per folder : n_blocks .csv with the
            # sentences.
            output_path = os.path.join(args.root_path,'run_experiment','Stimuli','visual',method, curr_subject)
            if not os.path.exists(output_path):
                    os.makedirs(output_path)
            file_name = os.path.join(output_path, (curr_subject + '_' + method  + '_' + str(curr_block) + '_' + '.csv' ))
            stimuli[curr_subject][curr_block].to_csv(file_name, sep='\t', encoding='utf-8')                
















