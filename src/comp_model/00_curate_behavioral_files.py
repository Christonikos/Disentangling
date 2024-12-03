#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read the behavioral files created with Pscychtoolbox when running the task 
and extract features of interest (i.e: violation, congruency etc).
Add those features as extra columns in the existing dataframe and store in 
/../data/$subject/curated
"""

# =============================================================================
# MODULES
# =============================================================================
import pandas as pd
import config as c

# =============================================================================
# LOAD DATA
# =============================================================================
def load_data(subject):
    print(f'Loading data for subject: {subject}')
    path2data=c.real(c.join('..','data',subject,'raw'))
    files=[pd.read_csv(c.join(path2data,f), sep='\t') for f in c.see(path2data)]
    dataset=pd.concat(files)
    ## CURATE THE DATAFRAME AND ADD FEATURES
    # drop the first column
    del dataset['Var1']
    # remove the NaNs (no responses from the subjects)
    # dataset = dataset[dataset['RT'].notna()]
    dataset['category']=dataset.condition.apply(lambda x: x.split('_')[-1])
    dataset['number']=dataset.condition.apply(lambda x: x.split('_')[-2])
    is_objrc=dataset.condition.apply(lambda x: x.split('_')[1]).apply(lambda y:'C' in y)
    dataset['structure']=''
    dataset['structure'][is_objrc]='obj'
    dataset['structure'][~is_objrc]='pp'
    dataset['condition']=dataset.condition.apply(lambda x: x.split('_')[0])
    # Now, add the main effects, violation, congruency and transition
    dataset['congruency']=''
    congruent = ['GSLS', 'GDLD']
    incongruent=['GSLD', 'GDLS']
    dataset['congruency'][dataset['condition'].isin(congruent)]='yes'
    dataset['congruency'][dataset['condition'].isin(incongruent)]='no'
    dataset['response']=''
    correct=['TP','TN']
    false=['FP','FN']
    dataset['response'][dataset['Behavioral'].isin(correct)]='correct'
    dataset['response'][dataset['Behavioral'].isin(false)]='false'    
    dataset['violation']=''
    dataset['violation'][dataset['violIndex']==1]='yes'
    dataset['violation'][dataset['violIndex']==0]='no'
    dataset = dataset.rename(columns={'category': 'feature'})
    dataset['feature']=dataset['feature'].replace({'syntactic': 'number', 'semantic': 'animacy',})
    dataset.number= dataset.number.map({'sing':'singular','plur':'plural'})
    dataset['number_attractor']=''
    dataset['number_attractor'][(dataset.structure=='pp') & (dataset.number=='plural')& (dataset.congruency=='no')&(dataset.feature=='number')]='singular'
    dataset['number_attractor'][(dataset.structure=='pp') & (dataset.number=='singular')& (dataset.congruency=='no')& (dataset.feature=='number')]='plural'
    dataset['number_attractor'][(dataset.structure=='obj') & (dataset.number=='singular')& (dataset.congruency=='no')& (dataset.feature=='number')]='singular'
    dataset['number_attractor'][(dataset.structure=='obj') & (dataset.number=='plural')& (dataset.congruency=='no')& (dataset.feature=='number')]='plural'
    dataset['animacy_attractor']=''
    dataset['animacy_attractor'][(dataset.feature=='animacy')&(dataset.congruency=='no')&(dataset.structure=='pp')&\
            (dataset['pair_index'].\
             apply(lambda x: x.split('-')[0])=='anim')]='inanimate'
    dataset['animacy_attractor'][(dataset.feature=='animacy')&(dataset.congruency=='no')&(dataset.structure=='pp')&\
            (dataset['pair_index'].\
             apply(lambda x: x.split('-')[0])=='inanim')]='animate'
    dataset[(dataset.structure=='obj')]['animacy_attractor']='none'
    
    
    dataset['embedded_noun_number']=''
    ## NUMBER FEATURE
    # ~~~~~~~~~~~~~~ PP ~~~~~~~~~~~~~~~~#
    dataset['embedded_noun_number'][(dataset.structure=='pp') & (dataset.number=='plural')& (dataset.congruency=='no')&(dataset.feature=='number')]='singular'
    dataset['embedded_noun_number'][(dataset.structure=='pp') & (dataset.number=='plural')& (dataset.congruency=='yes')&(dataset.feature=='number')]='plural'
    dataset['embedded_noun_number'][(dataset.structure=='pp') & (dataset.number=='singular')& (dataset.congruency=='no')&(dataset.feature=='number')]='plural'
    dataset['embedded_noun_number'][(dataset.structure=='pp') & (dataset.number=='singular')& (dataset.congruency=='yes')&(dataset.feature=='number')]='singular'
    # ~~~~~~~~~~~~~~ OBJ ~~~~~~~~~~~~~~~#
    dataset['embedded_noun_number'][(dataset.structure=='obj') & (dataset.number=='plural')& (dataset.congruency=='no')&(dataset.feature=='number')]='singular'
    dataset['embedded_noun_number'][(dataset.structure=='obj') & (dataset.number=='plural')& (dataset.congruency=='yes')&(dataset.feature=='number')]='plural'
    dataset['embedded_noun_number'][(dataset.structure=='obj') & (dataset.number=='singular')& (dataset.congruency=='no')&(dataset.feature=='number')]='plural'
    dataset['embedded_noun_number'][(dataset.structure=='obj') & (dataset.number=='singular')& (dataset.congruency=='yes')&(dataset.feature=='number')]='singular'
    
    ## ANIMACY FEATURE
    dataset['embedded_noun_number'][(dataset.structure=='pp') & (dataset.number=='plural')& (dataset.congruency=='no')&(dataset.feature=='animacy')]='plural'
    dataset['embedded_noun_number'][(dataset.structure=='pp') & (dataset.number=='singular')& (dataset.congruency=='no')&(dataset.feature=='animacy')]='singular'
    dataset['embedded_noun_number'][(dataset.structure=='pp') & (dataset.number=='singular')& (dataset.congruency=='yes')&(dataset.feature=='animacy')]='singular'
    dataset['embedded_noun_number'][(dataset.structure=='pp') & (dataset.number=='plural')& (dataset.congruency=='yes')&(dataset.feature=='animacy')]='plural'
    
    dataset['main_noun_animacy']=''
    dataset['main_noun_animacy'][(dataset['pair_index'].\
             apply(lambda x: x.split('-')[0])=='anim')]='animate'
    dataset['main_noun_animacy'][(dataset['pair_index'].\
             apply(lambda x: x.split('-')[0])=='inanim')]='inanimate'

    dataset['embedded_noun_animacy']=''
    dataset['embedded_noun_animacy'][(dataset['pair_index'].\
             apply(lambda x: x.split('-')[1])=='anim')]='animate'
    dataset['embedded_noun_animacy'][(dataset['pair_index'].\
             apply(lambda x: x.split('-')[1])=='inanim')]='inanimate'

    dataset.rename(columns={'number':'main_noun_number'}, inplace=True)
    dataset.drop(columns=['violation_type'], inplace=True)

    
    return dataset

def save_dataset(dataset, subject):
    path2folder=c.real(c.join('..','data',subject,'curated'))
    if not c.exists(path2folder):
        c.make(path2folder)
    fname=c.join(path2folder, f'stimuli_{subject}.csv')
    dataset.to_csv(fname)
    

# =============================================================================
# WRAP UP 
# =============================================================================
for subject in c.subjects_list:
    # read the dataset and add features as columns
    dataset=load_data(subject)
    # save the new dataset
    save_dataset(dataset, subject)