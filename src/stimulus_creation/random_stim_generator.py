#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 03:09:55 2019

@author: czacharo

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RANDOM STIMULUS GENERATOR
# Each cell in the sentence is randomly sampled
# from the lexicon. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""


# -----------------------------------------------------------------------------------#
# -------------------------#
# ---- IMPORT MODULES ----#
# -------------------------#
import os
import argparse
import random
from random import randrange
from lexicon import construct_lexicon 
import pandas as pd
# --------------------------------------------------------#


def gen_sent(condition, words, embedding, stim_type, number, pair_id, N, verbose):
    
    ###########################################################################
    # ~~~ INPUTS ~~~ #
    # 1. Condition: The base condition (One of "GSLS", "GSLD", "GDLS", "GDLD")
    # 2. words :    The lexicon created with the function lexicon.py
    # 3. embedding: Either pp or objrc
    # 4. stim_type: Syntax or Semantics
    # 5. number:    Grammatical number (singular or plural)
    # 6. pair_id:   The animacy pairs between n1 and n2
    # 7. verbose:   If true, the sentences are printed in the screen
    #               else, they are stored into a dataframe.
    
    # ~~~ OUTPUTS ~~~ #
    # Prints the sentences along with various corresponding features in a 
    # tabular form. You can run this script from the terminal (see the arg
    # parse inputs at the end of the script) and pipe the output to a .csv
    ###########################################################################    
        

    ########### INITIALIZE THE COLUMN NAMES #######################################
    sentences = pd.DataFrame()
    if verbose:
        print('\t'.join(['pair_index', \
             'n1' , 'pp', 'n2', 'v1','v2/n3/adverb', \
             'condition', 'sentence', 'violIndex', \
             'pp_freq' ,'n1_freq' , 'n2_freq', 'v1_freq','v2/n3/adverb_freq', \
             'pp_nl'  ,'n1_nl' , 'n2_nl', 'v1_nl','v2/n3/adverb_nl', 'violation_type']))
    else:
        sentences = pd.DataFrame(columns=['pair_index',\
                                          'n1','pp','n2',\
                                          'v1','v2/n3/adverb','condition',\
                                          'sentence','violIndex',\
                                          'pp_freq' ,'n1_freq' , 'n2_freq',\
                                          'v1_freq','v2/n3/adverb_freq','pp_nl',\
                                          'n1_nl' , 'n2_nl', 'v1_nl',\
                                          'v2/n3/adverb_nl', 'violation_type'])
        
        
    ################################################################################    

    ########### CONSTRUCT THE VOCABULARY ######################################
    # Vocabulary construction
    det             = 'the'
    pp              = words['prepositions']['pp']
    obj             = words['prepositions']['obj']
    humans          = words['nouns']['anim']['humans']
    vehicles        = words['nouns']['inan']['vehicles']
    preference      = words['verbs']['trans']['prefverbs']
    activities      = words['nouns']['activity']
    motion          = words['verbs']['intr']['unac']['motion']
    unergative      = words['verbs']['intr']['unerg']
#    honly_verbs     = words['verbs']['intr']['unac']['honly']
#    lonly_verbs     = words['verbs']['intr']['unac']['lonly']
#    adverbs_honk    = words['adverbs']['honk']
    adverbs_leak    = words['adverbs']['leak']
    veonly_verbs    = words['verbs']['intr']['unac']['vonly']
#    adverbs_accid   = words['adverbs']['accident']
    space           = ' ' 
    dot             = '.'    
    # ------------------------------------------------------------------------#    
        
    numbers  = ['sing', 'plur'] # used to create the conditions in the syntax
                                # block
     
    
    for _ in range(N): # number of sentences we wish to create
        ###########################################################
        ##################### SEMANT ##############################
        ###########################################################
        if stim_type == 'semantic':
            n1_number = number
            n2_number = number
            if condition == 'GSLS':
                sen_violIndex       = '0'
                ### RANDOM ANIMACY PAIRS ########################################
                if pair_id == 'random': # Choose randomnly the animacy
                    r_pair_id = random.choice(['anim-anim','inanim-inanim'])
                    animacy_index = r_pair_id
                    
                    if r_pair_id == 'anim-anim':
                        ## nouns #######
                        n1 = humans['neut'][n1_number]
                        n2 = humans['neut'][n2_number]
                        n3 = activities
                        ## verbs #########
                        v1 = preference[number]
                        v2 = unergative[number]
                        
                    elif r_pair_id== 'inanim-inanim':
                        ## nouns #######
                        n1 = vehicles[n1_number]
                        n2 = vehicles[n2_number]
                        ## verbs #########
                        v1 = random.choice([veonly_verbs,])
                        n3 = adverbs_leak
                        v1 = v1[number]
                        v2 = motion[number] 
                     
                
                ### ANIMATE-ANIMATE PAIRS ########################################
                elif pair_id == 'anim-anim':
                        ## nouns #######
                        n1 = humans['neut'][n1_number]
                        n2 = humans['neut'][n2_number]
                        n3 = activities
                        ## verbs #########
                        v1 = preference
                        animacy_index = pair_id
                
                elif pair_id== 'inanim-inanim':
                        ## nouns #######
                        n1 = vehicles[n1_number]
                        n2 = vehicles[n2_number]
                        ## verbs #########
                        v1 = random.choice([veonly_verbs])
                        n3 = adverbs_leak
                        v2 = motion[number]
                        animacy_index = pair_id
                        
            elif condition == 'GSLD':
                sen_violIndex       = '0'
                ### RANDOM ANIMACY PAIRS ########################################
                if pair_id == 'random': # Choose randomnly the animacy
                    r_pair_id = random.choice(['anim-inanim','inanim-anim'])
                    animacy_index = r_pair_id
                    
                    if r_pair_id == 'anim-inanim':
                        ## nouns #######
                        n1 = humans['neut'][n1_number]
                        n2 = vehicles[n2_number]
                        n3 = activities
                        ## verbs #########
                        v1 = preference[number]
                        v2 = unergative[number]
                        
                    elif r_pair_id== 'inanim-anim':
                        ## nouns #######
                        n1 = vehicles[n1_number]
                        n2 = humans['neut'][n2_number]
                        ## verbs #########
                        v1 = random.choice([veonly_verbs,])
                        n3 = adverbs_leak
                        v1 = v1[number]
                        v2 = motion[number]
                        
                elif pair_id == 'anim-inanim':
                        ## nouns #######
                        n1 = humans['neut'][n1_number]
                        n2 = vehicles[n2_number]
                        n3 = activities
                        ## verbs #########
                        v1 = preference[number]
                        v2 = unergative[number]
                        animacy_index = pair_id
                        
                elif pair_id== 'inanim-anim':
                        ## nouns #######
                        n1 = vehicles[n1_number]
                        n2 = humans['neut'][n1_number]
                        ## verbs #########
                        v1 = random.choice([veonly_verbs,])
                        n3 = adverbs_leak    
                        v1 = v1[number]
                        v2 = motion[number]
                        animacy_index = pair_id
                        
            elif condition == 'GDLS':
                sen_violIndex       = '1'
                ### RANDOM ANIMACY PAIRS ########################################
                if pair_id == 'random': # Choose randomnly the animacy
                    r_pair_id = random.choice(['anim-inanim','inanim-anim'])
                    animacy_index = r_pair_id  
                    
                    if r_pair_id == 'anim-inanim':
                        ## nouns #######
                        n1 = humans['neut'][n1_number]
                        n2 = vehicles[n2_number]
                        ## verbs #########
                        v1 = random.choice([veonly_verbs,])
                        n3 = adverbs_leak
                        v1 = v1[number]    
                        v2 = unergative[number] 
                              
                    
                    elif r_pair_id== 'inanim-anim':
                        ## nouns #######
                        n1 = vehicles[n1_number]
                        n2 = humans['neut'][n2_number]
                        n3 = activities
                        ## verbs #########
                        v1 = preference[number]
                        v2 = motion[number] 
                        
                        
                elif pair_id == 'anim-inanim':
                        ## nouns #######
                        n1 = humans['neut'][n1_number]
                        n2 = vehicles[n2_number]
                        ## verbs #########
                        v1 = random.choice([veonly_verbs,])
                        n3 = adverbs_leak
                        v1 = v1[number]    
                        v2 = unergative[number] 
                        animacy_index = pair_id
                        
                elif pair_id== 'inanim-anim':
                        ## nouns #######
                        n1 = vehicles[n1_number]
                        n2 = humans['neut'][n1_number]
                        n3 = activities
                        ## verbs #########
                        v1 = preference[number]
                        v2 = motion[number]
                        animacy_index = pair_id
                                        
                    
                    
            elif condition == 'GDLD':
                sen_violIndex       = '1'
                ### RANDOM ANIMACY PAIRS ########################################
                if pair_id == 'random': # Choose randomnly the animacy
                    r_pair_id = random.choice(['anim-anim','inanim-inanim'])
                    animacy_index = r_pair_id
                    
                    if r_pair_id == 'anim-anim':
                        ## nouns #######
                        n1 = humans['neut'][n1_number]
                        n2 = humans['neut'][n2_number]
                        ## verbs #########
                        v1 = random.choice([veonly_verbs,])
                        n3 = adverbs_leak
                        v1 = v1[number]
                        v2 = unergative[number] 
                        
                        
                    elif r_pair_id== 'inanim-inanim':
                        ## nouns #######
                        n1 = vehicles[n1_number]
                        n2 = vehicles[n1_number]
                        n3 = activities
                        ## verbs #########
                        v1 = preference[number]
                        v2 = motion[number] 
                
                elif pair_id == 'anim-anim':
                        ## nouns #######
                        n1 = humans['neut'][n1_number]
                        n2 = humans['neut'][n2_number]
                        ## verbs #########
                        v1 = random.choice([veonly_verbs,])
                        n3 = adverbs_leak
                        v1 = v1[number]    
                        v2 = unergative[number]         
                        animacy_index = pair_id
                        
                elif pair_id== 'inanim-inanim':
                        ## nouns #######
                        n1 = vehicles[n1_number]
                        n2 = vehicles[n2_number]
                        n3 = activities
                        ## verbs #########
                        v1 = preference[number]
                        v2 = motion[number] 
                        animacy_index = pair_id                        
                    
                    
                    
        ###########################################################
        ##################### SYNTAX ##############################
        ###########################################################
            
        elif stim_type == 'syntactic':
            s = numbers.index(number)
            animacy_index = 'anim-anim'
            
            if condition == 'GSLS':
                n1_number = number
                n2_number = number
                v1_number = number
                sen_violIndex       = '0'
                con_violIndex       = '0'
            
            elif condition  == 'GSLD':
                n1_number = number
                n2_number = numbers[1-s]
                v1_number = n1_number
                sen_violIndex       = '0'
                con_violIndex       = '1'

            elif condition  == "GDLS":
                n1_number = number
                n2_number = numbers[1-s]
                v1_number = numbers[1-s]
                sen_violIndex       = '1'
                con_violIndex       = '0'
            
            elif condition  == 'GDLD':
                n1_number = number
                n2_number = number
                v1_number = numbers[1-s]
                sen_violIndex       = '1'
                con_violIndex       = '1' 
                
            ## nouns #######
            n1 = humans['neut'][n1_number]
            n2 = humans['neut'][n2_number]
            n3 = activities
            ## verbs #########
            v1 = preference[v1_number]
            v2 = unergative[number]
                    
            
                
             
        ##################### N1 #########################################
        curr_n1 = {}
        for fid in ['s','f','n']:
            curr_n1[fid] = {}
        #----------------------------------------------------------------#
        random_index_n1 = randrange(len(n1['string']))
        curr_n1['s'] = n1['string'][random_index_n1] 
        curr_n1['f'] = n1['freqs'][random_index_n1] 
        curr_n1['n'] = n1['numletters'][random_index_n1] 
        #----------------------------------------------------------------#

        ##################### N2 #########################################
        curr_n2 = {}
        for fid in ['s','f','n']:
            curr_n2[fid] = {}
        #----------------------------------------------------------------#
        random_index_n2 = randrange(len(n2['string']))
        curr_n2['s'] = n2['string'][random_index_n2] 
        curr_n2['f'] = n2['freqs'][random_index_n2] 
        curr_n2['n'] = n2['numletters'][random_index_n2] 
        #----------------------------------------------------------------#
        
        
        ##################### V1 #########################################
        curr_v1 = {}
        for fid in ['s','f','n']:
            curr_v1[fid] = {}
        #----------------------------------------------------------------#
        random_index = randrange(len(v1['string']))
        curr_v1['s'] = v1['string'][random_index] 
        curr_v1['f'] = v1['freqs'][random_index] 
        curr_v1['n'] = v1['numletters'][random_index] 
        #----------------------------------------------------------------#
        
        
        ##################### V2 #########################################
        curr_v2 = {}
        for fid in ['s','f','n']:
            curr_v2[fid] = {}
        #----------------------------------------------------------------#
        random_index = randrange(len(v2['string']))
        curr_v2['s'] = v2['string'][random_index] 
        curr_v2['f'] = v2['freqs'][random_index] 
        curr_v2['n'] = v2['numletters'][random_index] 
        #----------------------------------------------------------------#
        
        
        
        ##################### N3 #########################################
        curr_n3 = {}
        for fid in ['s','f','n']:
            curr_n3[fid] = {}
        #----------------------------------------------------------------#
        random_index = randrange(len(n3['string']))
        curr_n3['s'] = n3['string'][random_index] 
        curr_n3['f'] = n3['freqs'][random_index] 
        curr_n3['n'] = n3['numletters'][random_index] 
        #----------------------------------------------------------------#
        
        ##################### PP #########################################
        curr_p = {}
        for fid in ['s','f','n']:
            curr_p[fid] = {}
        #----------------------------------------------------------------#
        if embedding == 'pp':
            random_index = randrange(len(pp['string']))
            # Prepositions only sampled from pp-fix that. 
            curr_p['s'] = pp['string'][random_index] 
            curr_p['f'] = pp['freqs'][random_index] 
            curr_p['n'] = pp['numletters'][random_index]
            
            n3_s=curr_n3['s']
            n3_f=curr_n3['f']
            n3_n=curr_n3['n']
            
        elif embedding == 'objrc':
            curr_p['s'] = obj['string'][0] 
            curr_p['f'] = obj['freqs'][0] 
            curr_p['n'] = obj['numletters'][0]
            
            n3_s=curr_v2['s']
            n3_f=curr_v2['f']
            n3_n=curr_v2['n']
            
            
        #----------------------------------------------------------------#
        
        if embedding == 'pp':
            condition_ID = condition + '_' + n1_number + '_' + stim_type
            curr_sentence = \
                        (det.capitalize() + space + curr_n1['s'] \
                        + space + str(curr_p['s'])  + ' the' \
                        + space +  curr_n2['s']  \
                        + space +  curr_v1['s'] \
                        + space +  curr_n3['s'] + dot ).strip()
        
        elif embedding == 'objrc':
            condition_ID = condition + '_C_' + n1_number + '_' + stim_type
            curr_sentence = \
                        (det.capitalize() + space + curr_n1['s'] \
                        + space + 'that'+ ' the' \
                        + space +  curr_n2['s']  \
                        + space +  curr_v1['s'] \
                        + space +  curr_v2['s'] + dot ).strip()
        
        ### SANITY CHECKS #####################################
        # 01: Check that the lemas are different
        if random_index_n1 == random_index_n2:
             continue
        else:
            if embedding == 'pp':
                violation_index = sen_violIndex
            elif embedding == 'objrc':
                violation_index = con_violIndex
            if verbose:
                print('\t'.join([animacy_index,\
                     curr_n1['s'] ,curr_p['s'], curr_n2['s'], curr_v1['s'], curr_n3['s'], \
                     condition_ID, curr_sentence, str(violation_index), \
                     str(curr_p['f']),\
                     str(curr_n1['f']) , str(curr_n2['f']), str(curr_v1['f']), str(curr_n3['f']), \
                     str(curr_p['n']),\
                     str(curr_n1['n']) , str(curr_n2['n']), str(curr_v1['n']), str(curr_n3['n']), \
                     stim_type ]))
            else:
                sentences.loc[_ ,'pair_index'] = animacy_index
                sentences.loc[_ ,'n1'] = curr_n1['s'] 
                sentences.loc[_ ,'pp'] = curr_p['s'] 
                sentences.loc[_ ,'n2'] = curr_n2['s']
                sentences.loc[_ ,'v1'] = curr_v1['s']
                sentences.loc[_ ,'v2/n3/adverb'] = n3_s 
                sentences.loc[_ ,'condition']    = condition_ID
                sentences.loc[_ ,'sentence']     = curr_sentence
                sentences.loc[_ ,'violIndex']  = str(violation_index)
                sentences.loc[_ ,'pp_freq']    = str(curr_p['f'])
                sentences.loc[_ ,'n1_freq']    = curr_n1['f']
                sentences.loc[_ ,'n2_freq']    = curr_n2['f']
                sentences.loc[_ ,'v1_freq']    = curr_v1['f']
                sentences.loc[_ ,'v2/n3/adverb_freq'] = n3_f 
                sentences.loc[_ ,'pp_nl'] = curr_p['n']
                sentences.loc[_ ,'n1_nl'] = curr_n1['n']
                sentences.loc[_ ,'n2_nl'] = curr_n2['n']
                sentences.loc[_ ,'v1_nl'] = curr_v1['n']
                sentences.loc[_ ,'v2/n3/adverb_nl'] = n3_n
                sentences.loc[_ ,'v1_freq']    = curr_v1['f']
                sentences.loc[_ ,'violation_type'] = stim_type
                
    return sentences
                
                
                
        



    
    
if __name__ == "__main__": 
    
    # --------------------------------------------------------#
    # -------------------------#
    # ---- PREFERENCES --------#
    # -------------------------#
    parser = argparse.ArgumentParser(description = 'Stimulus generator for English stimuli')
    parser.add_argument('-v', '--verbose', type=bool, default=0, help='visualization on/off')
    parser.add_argument('-p', '--root_path', type = str, default = os.path.join(
        os.sep,
        'volatile',
        'home',
        'czacharo',
        'Projects',
        'lang_lg',
        'Sources'
        ), help='The default rootpath')
    
    parser.add_argument('-c', '--condition', type=str, default='GDLD')
    parser.add_argument('-e', '--embedding', type=str, default='pp')
    parser.add_argument('-t', '--type', type=str, default='semantic')
    parser.add_argument('-a', '--animacy', type=str, default='random')
    parser.add_argument('-g', '--gram_num', type=str, default='sing')
    parser.add_argument('-N', '--number_of_sentences', type=int, default=10)
    # Get the default arguments
    args = parser.parse_args()
    # --------------------------------------------------------#

    # --------------------------------------------------------#
    # Load the lexicon
    words    = construct_lexicon(args.root_path, args.verbose)
    # --------------------------------------------------------#
    
    gen_sent(args.condition, words, args.embedding, args.type, args.gram_num, args.animacy, args.number_of_sentences, args.verbose)
    
    
    
    
    
    
    
    
    
    
    
    
    
    








    
    
    
    
    
    
    
