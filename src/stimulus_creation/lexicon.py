# This script creates the lexicon used at the English language experiment.
# This script can be run with  default parameters or you can specify
# those parameters (see INPUTS) from the command line.
#
# example : python3 lexicon.py --verbose=1
#
# INPUTS  :
#           1. verbose    : Boolean
#                                (If true, the user can view one of the created
#                                noun and verb dictionaries respectively)
#           2. root_path        : String
#                                (This is the path where the word-frequency list
#                                is stored)
#
# OUTPUTS :
#           1. Words      : Dictionary with keys :
#
#                               1. determinants
#                               2. nouns
#                               3. verbs
#
#                          The dictionary has the following form :
#                           Words
#                            ├── determinants
#                            │   └── definite_article
#                            ├── nouns
#                            │   ├── subj
#                            │   │   ├── sing
#                            │   │   │   ├── fem
#                            │   │   │   │   ├── freqs
#                            │   │   │   │   ├── numletters
#                            │   │   │   │   └── string
#                            │   │   │   ├── masc
#                            │   │   │   │   ├── freqs
#                            │   │   │   │   ├── numletters
#                            │   │   │   │   └── string
#                            │   │   │   └── neut
#                            │   │   │       ├── freqs
#                            │   │   │       ├── numletters
#                            │   │   │       └── string
#                            │   │   └── plur
#                            │   └──  obj
#                            │       ├── sing
#                            │       │   ├── fem
#                            │       │   │   ├── freqs
#                            │       │   │   ├── numletters
#                            │       │   │   └── string
#                            │       │   ├── masc
#                            │       │   │   ├── freqs
#                            │       │   │   ├── numletters
#                            │       │   │   └── string
#                            │       │   └── neut
#                            │       │       ├── freqs
#                            │       │       ├── numletters
#                            │       │       └── string
#                            │       └── plur
#                            └── verbs
#                                ├── intr
#                                │   ├── sing
#                                │   │   ├── numletters
#                                │   │   ├── string
#                                │   │   └── freqs
#                                │   └── plur
#                                └── trans
#                                    ├── sing
#                                    │   ├── numletters
#                                    │   ├── string
#                                    │   └── freqs
#                                    └── plur
#
#
#  Terminology :  subj = subject nouns, obj = object nouns, sing = singular,
#                 plur = plural, fem = feminine, masc = masculine, neut = neutral,
#                 intr = intransitive verbs, trans = transitive verbs.
#
# word-frequency list taken from : http://norvig.com/mayzner.html
# Written by : Christos Nikolaos Zacharopoulos and Yair Lakretz @UNICOG2019


# -----------------------------------------------------------------------------------#
# -------------------------#
# ---- IMPORT MODULES ----#
# -------------------------#

import os
import pandas as pd
import json
import argparse

def construct_lexicon(rootpath, verbose):
    # -------------------------#
    # ------ FILE INPUT ------#
    # -------------------------#
    # load the .csv that contains the word-frequency (space delimetered)
    uni_file_name = os.path.join(rootpath, 'stimuli_generator','Sources','unigram.csv')
    unigram = pd.read_csv(uni_file_name)

    # transform to a dictionary (lowercase the word entries)

    freqwords = dict(zip(unigram.word.str.lower(), unigram.frequency))

    # -------------------- CONSTRUCT LEXICON --------------------#
    # -------------------------#
    # ---- DETERMINERS -------#
    # -------------------------#

    determinants = {'definite_article': 'the'}

    # -------------------------#
    # ------- NOUNS -----------#
    # -------------------------#

    # ------------ CONSTRUCT THE WORD-LIST (NOUNS) ------------ #
    # sing = singular,   plur = plural
    # masc = masculine,  fem  = feminine
    # anim = animate,    inan = inanimate
    # ------------------------------------------------------------#

    ## ------------ ANIMATE NOUNS  ------------ ##

    #########################
    ## HUMANS ###############
    #########################

    # nouns - singular - masculine
    anim_humans_sing_masc_string = [
        'boy',
        'father',
        'husband',
        'brother',
        'man',
        'nephew']
    # nouns - singular - feminine
    anim_humans_sing_fem_string = [
        'girl',
        'mother',
        'wife',
        'sister',
        'woman',
        'niece']
 	#aunt
    # nouns - singular - neutral
    anim_humans_sing_neut_string = [
#        'girl',
#        'mother',
#        'wife',
#        'sister',
#        'woman',
#        'niece',
#        'boy',
#        'father',
#        'husband',
#        'brother',
#        'man',
#        'nephew',
        'athlete',
        'baker',
        'doctor',
        'farmer',
        'teacher',
        'lawyer',
        'actor',
	'author',
 	'banker',
#        'blogger',
        'barber',
        'chef',
        'dentist',
        'judge',
        'nurse',
        'painter',
        'pilot',
        'plumber',
        'tailor',
        'waiter',
        'vet',
#        'architect',
        'builder'
        ]

    # nouns - plural - masculine
    anim_humans_plur_masc_string = [
        'boys',
        'fathers',
        'husbands',
        'brothers',
        'men',
        'nephews']
    # nouns - plural - feminine
    anim_humans_plur_fem_string = [
        'girls',
        'mothers',
        'wives',
        'sisters',
        'women',
        'nieces']
    # nouns - plural - neutral
    anim_humans_plur_neut_string = [
#        'girls',
#        'mothers',
#        'wives',
#        'sisters',
#        'women',
#        'nieces',
#        'boys',
#        'fathers',
#        'husbands',
#        'brothers',
#        'men',
#        'nephews',
        'athletes',
        'bakers',
        'doctors',
        'farmers',
        'teachers',
        'lawyers',
        'actors',
	'authors',
 	'bankers',
#        'bloggers',
        'barbers',
        'chefs',
        'dentists',
        'judges',
        'nurses',
        'painters',
        'pilots',
        'plumbers',
        'tailors',
        'waiters',
        'vets',
#        'architects',
        'builders'
        ]

    #########################
    ## ANIMALS###############
    #########################

    anim_animals_sing_string = [
        'elephant',
        'giraffe',
        'bear',
        'rhino',
        'zebra',
        'panda',
        'camel',
        'kangaroo',
        'lion',
        'monkey',
        'pig',
        'tiger'
        ]
    



    anim_animals_plur_string = [
        'elephants',
        'giraffes',
        'bears',
        'rhinos',
        'zebras',
        'pandas',
        'camels',
        'kangaroos',
        'lions',
        'monkeys',
        'pigs',
        'tigers'
        ]
    
    ## ------------ INANIMATE NOUNS  ------------ ##

    #########################
    ## VEHICLES #############
    #########################

    inan_vehicles_sing_string = [
        'car',
        'bus', 
        'taxi',
        'truck',
#        'jeep',
        'tractor',
#        'camper',
        'scooter',
        'van',
#        'ambulance',
#        'bulldozer',
#        'crane'
        ]

    inan_vehicles_plur_string = [
        'cars',
        'buses',
        'taxis',
        'trucks',
#        'jeeps',
        'tractors',
#        'campers',
        'scooters',
        'vans',
#        'ambulances',
#        'bulldozers',
#        'cranes'
        ]

    #########################
    ## DOMESTIC OBJECTS######
    #########################

    inan_domobj_sing_string = [
        'carpet',
        'sofa',
        'chair',
        'table',
        'pillow',
        'curtain']

    inan_domobj_plur_string = [
        'carpets',
        'sofas',
        'chairs',
        'tables',
        'pillows',
        'curtains']





    ## ------------ ACTIVITIES ------------ ##
    activity_string = [
        'climbing',
        'skiing',
        'cooking',
        'shopping',
        'painting',
        'studying',
        'walking',
        'cycling',
        'farming',
        'fencing',
        'gambling',
        'knitting',
        'acting',
        'boxing',
        'bowling',
        'camping',
        'fishing',
        'skating',
        'dancing',
        'sailing',
        'yachting',
        'hunting',
        'spinning',
        'driving']


    ## ------------- DISCILPINES --------------------- ##
    discipline_string = [
                    'humility',
                    'beauty',
                    'honor',
                    'honesty',
                    'gratitude',
                    'loyalty',
                    'modesty']
    
    ## ------------- LANGUAGES --------------------- ##
    languages_string = ['english',
                        'spanish',
                        'arabic',
                        'russian',
                        'french',
                        'japanese']


    # ----------------------------------------------------------#
    # CREATE THE NOUNS DICTIONARY

    nouns = {}
    # 1st level : animacy
    for syntaxID in ['anim', 'inan', 'activity','discipline','languages']:  # syntax ID
        nouns[syntaxID] = {}
        # split here depending on the animacy
        if syntaxID == 'anim':
            # 2nd level: semantic category
            for sem_cat in ['humans', 'animals']:
                nouns[syntaxID][sem_cat] = {}
                if sem_cat == 'humans': 
                        # 3rd level: gender
                        for genderID in ['masc', 'fem', 'neut']:  # gender ID
                            nouns[syntaxID][sem_cat][genderID] = {}
                            for gramm_nID in ['sing', 'plur']:  # grammatical number ID
                                nouns[syntaxID][sem_cat][genderID][gramm_nID] = {}
                                # WORDS --> DICTIONARY
                                nouns[syntaxID][sem_cat][genderID][gramm_nID]['string'] = \
                                    eval(syntaxID + '_' + sem_cat + '_' +
                                        gramm_nID + '_' + genderID + '_string')
                                #########################
                                ## FEATURES #############
                                #########################
                                for featureID in ['freqs', 'numletters']:
                                    nouns[syntaxID][sem_cat][genderID][gramm_nID][featureID] = [
                                    ]
                                # 1. WORD-FREQ FEATURE
                                    if featureID == 'freqs':
                                        for wordID in range(len(nouns[syntaxID][sem_cat][genderID][gramm_nID]['string'])):
                                            word_freq = [
                                                freqwords[nouns[syntaxID][sem_cat][genderID][gramm_nID]['string'][wordID]]]
                                            # transform the list of integers to string and update the feature key of the dict
                                            nouns[syntaxID][sem_cat][genderID][gramm_nID][featureID].append(
                                                ''.join(str(e) for e in word_freq))
                                # 2. NUM LETTERS FEATURE
                                    elif featureID == 'numletters':
                                        for wordID in range(len(nouns[syntaxID][sem_cat][genderID][gramm_nID]['string'])):
                                            nouns[syntaxID][sem_cat][genderID][gramm_nID][featureID].append(
                                                len(nouns[syntaxID][sem_cat][genderID][gramm_nID]['string'][wordID]))
                elif sem_cat == 'animals':                           
                        for gramm_nID in ['sing', 'plur']:  # grammatical number ID
                                nouns[syntaxID][sem_cat][gramm_nID] = {}
                                # WORDS --> DICTIONARY
                                nouns[syntaxID][sem_cat][gramm_nID]['string'] = \
                                    eval(syntaxID + '_' + sem_cat + '_' +
                                        gramm_nID + '_string')
                                #########################
                                ## FEATURES #############
                                #########################
                                for featureID in ['freqs', 'numletters']:
                                    nouns[syntaxID][sem_cat][gramm_nID][featureID] = []
                                # 1. WORD-FREQ FEATURE
                                    if featureID == 'freqs':
                                        for wordID in range(len(nouns[syntaxID][sem_cat][gramm_nID]['string'])):
                                            word_freq = [
                                                freqwords[nouns[syntaxID][sem_cat][gramm_nID]['string'][wordID]]]
                                            # transform the list of integers to string and update the feature key of the dict
                                            nouns[syntaxID][sem_cat][gramm_nID][featureID].append(''.join(str(e) for e in word_freq))
                                # 2. NUM LETTERS FEATURE
                                    elif featureID == 'numletters':
                                        for wordID in range(len(nouns[syntaxID][sem_cat][gramm_nID]['string'])):
                                            nouns[syntaxID][sem_cat][gramm_nID][featureID].append(
                                                len(nouns[syntaxID][sem_cat][gramm_nID]['string'][wordID]))

        elif syntaxID == 'inan':
            for sem_cat in ['vehicles', 'domobj']:
                nouns[syntaxID][sem_cat] = {}
                for gramm_nID in ['sing', 'plur']:  # grammatical number ID
                    nouns[syntaxID][sem_cat][gramm_nID] = {}
                    # WORDS --> DICTIONARY
                    nouns[syntaxID][sem_cat][gramm_nID]['string'] = \
                        eval(syntaxID + '_' + sem_cat +
                            '_' + gramm_nID + '_string')
                    #########################
                    ## FEATURES #############
                    #########################
                    for featureID in ['freqs', 'numletters']:
                        nouns[syntaxID][sem_cat][gramm_nID][featureID] = []
                    # 1. WORD-FREQ FEATURE
                        if featureID == 'freqs':
                            for wordID in range(len(nouns[syntaxID][sem_cat][gramm_nID]['string'])):
                                word_freq = [
                                    freqwords[nouns[syntaxID][sem_cat][gramm_nID]['string'][wordID]]]
                                # transform the list of integers to string and update the feature key of the dict
                                nouns[syntaxID][sem_cat][gramm_nID][featureID].append(
                                    ''.join(str(e) for e in word_freq))
                    # 2. NUM LETTERS FEATURE
                        elif featureID == 'numletters':
                            for wordID in range(len(nouns[syntaxID][sem_cat][gramm_nID]['string'])):
                                nouns[syntaxID][sem_cat][gramm_nID][featureID].append(
                                    len(nouns[syntaxID][sem_cat][gramm_nID]['string'][wordID]))
        elif syntaxID == 'activity' or syntaxID == 'discipline' or syntaxID == 'languages':
            # WORDS --> DICTIONARY
            nouns[syntaxID]['string'] = \
                eval(syntaxID + '_string')
            #########################
            ## FEATURES #############
            #########################
            for featureID in ['freqs', 'numletters']:
                nouns[syntaxID][featureID] = []
            # 1. WORD-FREQ FEATURE
                if featureID == 'freqs':
                    for wordID in range(len(nouns[syntaxID]['string'])):
                        word_freq = [
                            freqwords[nouns[syntaxID]['string'][wordID]]]
                        # transform the list of integers to string and update the feature key of the dict
                        nouns[syntaxID][featureID].append(
                            ''.join(str(e) for e in word_freq))
            # 2. NUM LETTERS FEATURE
                elif featureID == 'numletters':
                    for wordID in range(len(nouns[syntaxID]['string'])):
                        nouns[syntaxID][featureID].append(
                            len(nouns[syntaxID]['string'][wordID]))


    # visualize the created noun dictionary
    if verbose:
        print(json.dumps(nouns, indent=4))

    # -------------------------#
    # -------- VERBS ----------#
    # -------------------------#

    # ------------ CONSTRUCT THE WORD-LIST (VERBS) ------------ #
    # intr = intransitive verbs, tran = transitive verbs

    #########################
    ## INTRANSITIVE VERBS ##
    #########################

    #-----------------------#
    #-UNACCUSATIVE VERBS ---#
    #-----------------------#

    # ---- MOTION VERBS ----#
    intr_unac_motion_sing_string = [
#        'arrives',
        'departs',
        'moves',
        #'leaves',
        'turns',
        'stops',
        # 'revolves',
        # 'drifts',
        # 'glides',
        # 'slides',
        # 'spins',
        # 'turns'
    ]

    intr_unac_motion_plur_string = [
#        'arrive',
        'depart',
        'move',
        #'leave',
        'turn',
        'stop',
        # 'revolve',
        # 'drift',
        # 'glide',
        # 'slide',
        # 'spin',
        # 'turn'
    ]


    # ---- ANIMAL SOUNDS VERBS ----#
    intr_unac_ansounds_sing_string = [
        'barks',
        'growls',
        'moos',
        'quacks',
        'squeaks',
        'howls'
    ]
    intr_unac_ansounds_plur_string = [
        'bark',
        'growl',
        'moos',
        'quack',
        'squeak',
        'howl'
    ]
    
    
    # ---- CONSTRUCTION VERBS ---- #
    intr_unac_consverbs_sing_string = [
        'grills',
        'builds',
        'scrambles',
        'stirs',
        'blends',
        'crops',
        'boils']

    intr_unac_consverbs_plur_string = [
        'grill',
        'build',
        'scramble',
        'stir',
        'blend',
        'crop',
        'boil']


    # ---- VEHICLE SPECIFIC VERBS ----#
    intr_unac_vonly_sing_string = [
#                    'brakes',
#                    'skids',
#                    'crashes',
#                    'smashes'
                    'leaks',
                    'rusts',
                    'malfunctions'
                    ]
    intr_unac_vonly_plur_string = [
#                    'brake',
#                    'skid',
#                    'crash',
#                    'smash'
                    'leak',
                    'rust',
                    'malfunction'
                    ]
    intr_unac_honly_sing_string = [
                    'honks']
    intr_unac_honly_plur_string = [
                    'honk']
    
    intr_unac_lonly_sing_string = [
                    'leaks']
    intr_unac_lonly_plur_string = [
                    'leak']    
    
    # ---- ANIMAL SPECIFIC VERBS ----#
    intr_unac_aonly_sing_string = [
                    'bites',
                    'licks',
                    'nudges'
#                    'eats'
                    ]
    intr_unac_aonly_plur_string = [
                    'bite',
                    'lick',
                    'nudge'
                    ]


    # ---- VERBS DESCRIBING A NON-VOLUNTARY EMISSION
    # OF STIMULI THAT HAS AN IMPACT ON THE SENSES ---- #
    intr_unac_stimemission_sing_string = [
        'shines',
        'sparkles',
        'glitters',
        'glows',
        'smells',
        'stinks']

    intr_unac_stimemission_plur_string = [
        'shine',
        'sparkle',
        'glitter',
        'glow',
        'smell',
        'stink']

    #-----------------------#
    #-UNERGATIVE VERBS------#
    #-----------------------#
    intr_unerg_sing_string = [
#        'smiles',
#        'cries',
#        'laughs',
        'prays',
#        'coughs',
#        'sneezes',
        'sits',
#        'runs',
#        'swims',
#        'lies',
        'dies',
#        'studies',
        'arrives',
#        'moves',
#        'leaves',
#        'turns'
        ]

    intr_unerg_plur_string = [
#        'smile',
#        'cry',
#        'laugh',
        'pray',
#        'cough',
#        'sneeze',
        'sit',
#        'run',
#        'swim',
#        'lie',
        'die',
#        'study',
        'arrive',
#        'move',
#        'leave',
#        'turn'
        ]


    #########################
    ## TRANSITIVE VERBS #####
    #########################

    # ---- PREFERENCE VERBS ----#

    trans_prefverbs_sing_string = [
        'likes',
        'loves',
        'hates',
        'avoids',
        'dislikes',
        'fears',
        'prefers',
        # 'abhors',
        # 'avoids',
         'detests',
        # 'dreads',
        # 'evades',
        # 'fancies'
    ]

    trans_prefverbs_plur_string = [
        'like',
        'love',
        'hate',
        'avoid',
        'dislike',
        'fear',
        'prefer',
        # 'abhor',
        # 'avoid',
         'detest',
        # 'dread',
        # 'evade',
        # 'fancy'
    ]
    
    trans_knowverbs_sing_string = [
                  'knows',
#                  'recognises'
                  ]
    trans_knowverbs_plur_string = [
                  'know',
#                  'recognise'
                  ]

    trans_admirverbs_sing_string = [
                    'admires']
    
    trans_admirverbs_plur_string = [
                    'admire']
    trans_pushpull_sing_string = [
                    'pushes',
                    'pulls']
    trans_pushpull_plur_string = [
                    'push',
                    'pull']


    # ----------------------------------------------------------#
    # VERBS DICTIONARY INITIALIZATION
    # trans = transitive, intr = intransitive

    verbs = {}
    # 1st level : transitivity
    for verbID in ['intr', 'trans']:  # verb category ID
        verbs[verbID] = {}
        # split depending on the verb category
        if verbID == 'intr':
            # 2nd level: Intransitive verbs subcategories (unaccusative & unergative)
            for intr_cat in ['unac', 'unerg']:
                verbs[verbID][intr_cat] = {}
                # 3rd level: within sub-categories separation (motion verbs etc)
                if intr_cat == 'unac':
                        # sub categories of unaccusative verbs
                    for unacID in ['motion', 'stimemission', 'consverbs', 'ansounds', 'vonly', 'aonly','honly', 'lonly']:
                        verbs[verbID][intr_cat][unacID] = {}
                        for gramm_nID in ['sing', 'plur']:  # grammatical number ID
                            verbs[verbID][intr_cat][unacID][gramm_nID] = {}
                            # WORDS --> DICTIONARY
                            verbs[verbID][intr_cat][unacID][gramm_nID]['string'] = \
                                eval(verbID + '_' + intr_cat + '_' +
                                    unacID + '_' + gramm_nID + '_string')
                            #########################
                            ## FEATURES #############
                            #########################
                            for featureID in ['freqs', 'numletters']:
                                verbs[verbID][intr_cat][unacID][gramm_nID][featureID] = [
                                ]
                            # 1. WORD-FREQ FEATURE
                                if featureID == 'freqs':
                                    for wordID in range(len(verbs[verbID][intr_cat][unacID][gramm_nID]['string'])):
                                            try:
                                                word_freq = [
                                                    freqwords[verbs[verbID][intr_cat][unacID][gramm_nID]['string'][wordID]]]
                                                # transform the list of integers to string and update the feature key of the dict
                                                verbs[verbID][intr_cat][unacID][gramm_nID][featureID].append(
                                                    ''.join(str(e) for e in word_freq))
                                            except:
                                                    
                                                    verbs[verbID][intr_cat][unacID][gramm_nID][featureID].append('0')
                            # 2. NUM LETTERS FEATURE
                                elif featureID == 'numletters':
                                    for wordID in range(len(verbs[verbID][intr_cat][unacID][gramm_nID]['string'])):
                                        verbs[verbID][intr_cat][unacID][gramm_nID][featureID].append(
                                            len(verbs[verbID][intr_cat][unacID][gramm_nID]['string'][wordID]))
                elif intr_cat == 'unerg':
                    for gramm_nID in ['sing', 'plur']:  # grammatical number ID
                        verbs[verbID][intr_cat][gramm_nID] = {}
                        # WORDS --> DICTIONARY
                        verbs[verbID][intr_cat][gramm_nID]['string'] = \
                            eval(verbID + '_' + intr_cat +
                                '_' + gramm_nID + '_string')
                        #########################
                        ## FEATURES #############
                        #########################
                        for featureID in ['freqs', 'numletters']:
                            verbs[verbID][intr_cat][gramm_nID][featureID] = [
                            ]
                        # 1. WORD-FREQ FEATURE
                            if featureID == 'freqs':
                                for wordID in range(len(verbs[verbID][intr_cat][gramm_nID]['string'])):
                                    word_freq = [
                                        freqwords[verbs[verbID][intr_cat][gramm_nID]['string'][wordID]]]
                                    # transform the list of integers to string and update the feature key of the dict
                                    verbs[verbID][intr_cat][gramm_nID][featureID].append(
                                        ''.join(str(e) for e in word_freq))
                        # 2. NUM LETTERS FEATURE
                            elif featureID == 'numletters':
                                for wordID in range(len(verbs[verbID][intr_cat][gramm_nID]['string'])):
                                    verbs[verbID][intr_cat][gramm_nID][featureID].append(
                                        len(verbs[verbID][intr_cat][gramm_nID]['string'][wordID]))
        elif verbID == 'trans':
            for trans_cat in ['prefverbs','admirverbs','knowverbs','pushpull']:
                verbs[verbID][trans_cat] = {}
                for gramm_nID in ['sing', 'plur']:  # grammatical number ID
                        verbs[verbID][trans_cat][gramm_nID] = {}
                        # WORDS --> DICTIONARY
                        verbs[verbID][trans_cat][gramm_nID]['string'] = \
                            eval(verbID + '_' + trans_cat +
                                '_' + gramm_nID + '_string')
                        #########################
                        ## FEATURES #############
                        #########################
                        for featureID in ['freqs', 'numletters']:
                            verbs[verbID][trans_cat][gramm_nID][featureID] = [
                            ]
                        # 1. WORD-FREQ FEATURE
                            if featureID == 'freqs':
                                for wordID in range(len(verbs[verbID][trans_cat][gramm_nID]['string'])):
                                    word_freq = [
                                        freqwords[verbs[verbID][trans_cat][gramm_nID]['string'][wordID]]]
                                    # transform the list of integers to string and update the feature key of the dict
                                    verbs[verbID][trans_cat][gramm_nID][featureID].append(
                                        ''.join(str(e) for e in word_freq))
                        # 2. NUM LETTERS FEATURE
                            elif featureID == 'numletters':
                                for wordID in range(len(verbs[verbID][trans_cat][gramm_nID]['string'])):
                                    verbs[verbID][trans_cat][gramm_nID][featureID].append(
                                        len(verbs[verbID][trans_cat][gramm_nID]['string'][wordID]))
        

    # visualize the created verb dictionary
    if verbose:
        print(json.dumps(verbs, indent=4))


    # -------------------------------#
    # -------- ADVERBS --------------#
    # -------------------------------#
    adv_state_string = [
        'playfully', 
        #'lovingly', 
        'charmingly',
        #'angrily',
        #'furiously',
        'cheerfully']


    adv_speed_string = [
          #  'unsafely',
            'dangerously',
          #  'unsecurely',
            'recklessly',
            'carelessly',
          #  'sloppily',        
            ]
    


    adv_accident_string = [
        'loudly',
        'noisily',
#        'violently',
#        'dangerously',
        'abruptly',
        'suddenly',
#        'unexpectedly'
] 
	
    adv_honk_string = [
                    'loudly',
                    'noisily',
                    'twice',
                    'once',
#                    'repeatedly',
                    ]

    adv_leak_string = [
                        'abruptly',
                        'suddenly',
#			'quickly',	
#                        'twice',
#                        'once',
    #                    'repeatedly',
                        ]


    # ----------------------------------------------------------#
    # CREATE THE ADJ DICTIONARY
    adv = {}
    for advID in ['accident', 'honk', 'leak']:
        adv[advID] = {}
        # WORDS --> DICTIONARY
        adv[advID]['string'] = eval('adv_' + advID + '_string')
        #########################
        ## FEATURES #############
        #########################
        for featureID in ['freqs', 'numletters']:
            adv[advID][featureID] = []
            # 1. WORD-FREQ FEATURE
            if featureID == 'freqs':
                for wordID in range(len(adv[advID]['string'])):
                    word_freq = [
                        freqwords[adv[advID]['string'][wordID]]]
                    # transform the list of integers to string and update the feature key of the dict
                    adv[advID][featureID].append(
                        ''.join(str(e) for e in word_freq))
                    # 2. NUM LETTERS FEATURE
            elif featureID == 'numletters':
                for wordID in range(len(adv[advID]['string'])):
                    adv[advID][featureID].append(
                        len(adv[advID]['string'][wordID]))


    # ----------------------------------------------------------#
    # CREATE THE PREPOSITIONS DICTIONARY
    # -------------------------------#
    # -------- PREPOSITIONS -----------#
    # -------------------------------#
    prep_pp_string = [
        'near',
        'by',
        'beside']
    prep_obj_string = [
        'that',
#        'which'
]
    prep = {}
    for prepID in range(len(prep_pp_string)):
        for prepID in ['pp', 'obj']:
            prep[prepID] = {}
            # WORDS --> DICTIONARY
            prep[prepID]['string'] = eval('prep_' + prepID + '_string')
            #########################
            ## FEATURES #############
            #########################
            for featureID in ['freqs', 'numletters']:
                prep[prepID][featureID] = []
                # 1. WORD-FREQ FEATURE
                if featureID == 'freqs':
                    for wordID in range(len(prep[prepID]['string'])):
                        word_freq = [
                            freqwords[prep[prepID]['string'][wordID]]]
                        # transform the list of integers to string and update the feature key of the dict
                        prep[prepID][featureID].append(
                            ''.join(str(e) for e in word_freq))
                        # 2. NUM LETTERS FEATURE
                elif featureID == 'numletters':
                    for wordID in range(len(prep[prepID]['string'])):
                        prep[prepID][featureID].append(
                            len(prep[prepID]['string'][wordID]))


    # -------------------------#
    # ------- OUTPUT ----------#
    # -------------------------#

    # Collect everything to a dictionary
    words = {'determinants': determinants.copy(), 'nouns': nouns.copy(),
            'verbs': verbs.copy(), 'prepositions': prep.copy(), 'adverbs': adv.copy()}

    if verbose:
        '''
        import visualisedictionary as vd
        # Visualize the Verbs dict
        V = vd.KeysGraph(verbs)
        V.draw('verbs.png')
        # Visualize the Nouns dict
        N = vd.KeysGraph(nouns)
        N.draw('nouns.png')
        # Visualize the Words dict
        W = vd.KeysGraph(Words)
        W.draw('words.png')
        '''
        from lolviz import objviz
        objviz(Words)
        objviz(verbs)
        objviz(nouns)
        objviz(prep)
        objviz(adv)
    
    return words

if __name__ == "__main__":
    # -------------------------#
    # ---- PREFERENCES -------#
    # -------------------------#

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', type=bool, default=0,
                        help='View the created dictionaries')
    parser.add_argument('--root_path', type=str, default=os.path.join(
        os.sep,
        'volatile',
        'home',
        'czacharo',
        'Projects',
        'lang_lg',
        'Sources'
    ), help='The default rootpath')

    # Get the default arguments
    args = parser.parse_args()
    
    words = construct_lexicon(args.root_path, args.verbose)
