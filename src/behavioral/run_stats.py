#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read the .csv created by "calculate_error_rate.py" and run statistics on it.

Created on Mon Sep 20 13:23:35 2021

@author: cz257680
"""
# =============================================================================
# MODULES
# =============================================================================
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import config as c







# =============================================================================
# EFFECTS
# =============================================================================
    

def get_error_and_sem(data):
    
    error, sem, std=({} for i in range(0,3))
    for v in ['grammatical','violation']:
        error[v]={}
        sem[v]={}
        std[v]={}
        for con in ['congruent','incongruent']:
            error[v][con]={}
            sem[v][con]={}
            std[v][con]={}
        
            
    ## ERROR    
    error['violation']['congruent']=\
        data[(data.violation==1)&(data.congruency==1)].\
            error_per_combination.mean()
    error['violation']['incongruent']=\
        data[(data.violation==1)&(data.congruency==0)].\
            error_per_combination.mean()
    error['grammatical']['congruent']=\
        data[(data.violation==0)&(data.congruency==1)].\
            error_per_combination.mean()                    
    error['grammatical']['incongruent']=\
        data[(data.violation==0)&(data.congruency==0)].\
            error_per_combination.mean()     

    ## SEM
    sem['violation']['congruent']=\
        data[(data.violation==1)&(data.congruency==1)].\
            error_per_combination.sem()
    sem['violation']['incongruent']=\
        data[(data.violation==1)&(data.congruency==0)].\
            error_per_combination.sem()
    sem['grammatical']['congruent']=\
        data[(data.violation==0)&(data.congruency==1)].\
            error_per_combination.sem()                    
    sem['grammatical']['incongruent']=\
        data[(data.violation==0)&(data.congruency==0)].\
            error_per_combination.sem()     


    ## STD
    std['violation']['congruent']=\
        data[(data.violation==1)&(data.congruency==1)].\
            error_per_combination.std()
    std['violation']['incongruent']=\
        data[(data.violation==1)&(data.congruency==0)].\
            error_per_combination.std()
    std['grammatical']['congruent']=\
        data[(data.violation==0)&(data.congruency==1)].\
            error_per_combination.std()                    
    std['grammatical']['incongruent']=\
        data[(data.violation==0)&(data.congruency==0)].\
            error_per_combination.std()     



    return error, sem, std


# =============================================================================
# PLOT 
# =============================================================================
def plot_interaction_plots(data):
    fig=plt.figure(dpi=100, facecolor='w', edgecolor='w')
    fig.set_size_inches(12, 4)
    
    labels=['Congruent', 'Incongruent']

    lines=[]
    for idx, construction in enumerate(c.constructions):
        if construction=='pp_syntax':
            structure='pp'
            feature='number'
            title=r'$\mathcal{PP-Number}$'
        elif construction=='objrc_syntax':
            structure='obj'
            feature='number'
            title=r'$\mathcal{ObjRC-Number}$'
        elif construction=='pp_semantics':
            structure='pp'
            feature='animacy'
            title=r'$\mathcal{PP-Animacy}$'
            
        # parse the data
        q=f"structure=='{structure}' & feature=='{feature}'"
        df=data.query(q)
  
        error,sem, std=get_error_and_sem(df)

 
        
        mrsize=12
        x1,x2=1,2
       
        plt.subplot(1,3,idx+1)
        # Ghost data
        l1=plt.plot(x1, error['grammatical']['congruent'], c="gray", marker="o",
                markersize=mrsize,markeredgecolor='k', label='congruent') 
        
        lines.append(l1[0])
        plt.plot(x1, error['grammatical']['congruent'], c="g", marker="o",
                markersize=mrsize,markeredgecolor='k')  
        plt.errorbar(x1, error['grammatical']['congruent'],
                    yerr=sem['grammatical']['congruent'],
                    c="g", uplims=False, lolims=False,)
       
        plt.plot(x1, error['grammatical']['incongruent'], c="g", marker="X",
                markersize=mrsize,markeredgecolor='k') 
        plt.errorbar(x1, error['grammatical']['incongruent'],
                    yerr=sem['grammatical']['incongruent'],
                    c="g", uplims=False, lolims=False,)
       
       
        plt.plot(x2, error['violation']['congruent'], c="r", marker="o", 
                markersize=mrsize,markeredgecolor='k')  
        plt.errorbar(x2, error['violation']['congruent'],
                    yerr=sem['violation']['congruent'],
                    c="r", uplims=False, lolims=False,)
       
        # Ghost data
        l2=plt.plot(x2, error['violation']['incongruent'], c="gray", marker="X", 
                markersize=mrsize,markeredgecolor='k') 
        lines.append(l2[0])
        
        plt.plot(x2, error['violation']['incongruent'], c="r", marker="X", 
                markersize=mrsize,markeredgecolor='k') 
        plt.errorbar(x2, error['violation']['incongruent'],
                    yerr=sem['violation']['incongruent'],
                    c="r", uplims=False, lolims=False,) 
       
        plt.plot([x1,x2], [error['grammatical']['incongruent'],
                          error['violation']['incongruent']],'k--', 
                     linewidth=1.2, zorder=1)
       
        plt.plot([x1,x2], [error['grammatical']['congruent'],
                          error['violation']['congruent']],'k-', 
                     linewidth=1.2, zorder=1)
        if idx==0:
            plt.ylabel(r'%Error', fontsize=1.2*mrsize)
        plt.xticks([x1,x2], ['Grammatical','Violation'], 
                       style='oblique', fontweight='bold')
        plt.xlim([x1-.2, x2+.2])
        plt.ylim([0,50])
        # plt.title(title,style='oblique', fontweight='bold', fontsize=1*mrsize)
        sns.despine( offset=10)
        
    # plt.figlegend( lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04),
    #        ncol=2, labelspacing=0, fancybox=True,
    #        facecolor='ghostwhite', edgecolor='silver' 
    #       )

        
    fig.tight_layout(pad=2)
    plt.savefig(fname='main_effects.png',bbox_inches='tight', dpi=1200)
    plt.show()


# %%
# =============================================================================
# LOAD DATA 
# =============================================================================
data=pd.read_csv('data.csv')


# =============================================================================
# PLOT THE INTERACTION PLOTS
# =============================================================================

plot_interaction_plots(data)




# =============================================================================
# STATS
# =============================================================================
from bioinfokit.analys import stat
from tabulate import tabulate
## Get the results per construction
for construction in ['pp_number','obj_number','pp_animacy']:
    if construction=='pp_number':
        feature='number'
        structure='pp'
    elif construction=='obj_number':        
        feature='number'
        structure='obj'
    elif construction=='pp_animacy':        
        feature='animacy'
        structure='pp'        
    # query the data
    df=data[(data.structure==f'{structure}')&(data.feature==f'{feature}')]

    # Now, split further based on the desired interaction (congruency VS interference)
    print(2*'\n',20*'*',f'{construction}', 20*'*')
    
    for effect in ['congruency','linear_interference']:

        model = f'error_per_construction~ C(violation)*C({effect})'
        
        res = stat()
        res.anova_stat(df=df, anova_model=model)
        table=res.anova_summary
        # create a new df to store results
        table=table.iloc[:-1]
        new=pd.DataFrame(columns=['F','p_val'])
        new.F=table.F
        new.p_val=table['PR(>F)']
        
        print(tabulate(new, headers='keys', tablefmt='psql'))       





# =============================================================================
# STATS (MEAN +\- STD for the Paper)
# =============================================================================

def get_summary_statistics(data):
    for idx, construction in enumerate(c.constructions):
        if construction=='pp_syntax':
            structure='pp'
            feature='number'
            title=r'$\mathcal{PP-Number}$'
        elif construction=='objrc_syntax':
            structure='obj'
            feature='number'
            title=r'$\mathcal{ObjRC-Number}$'
        elif construction=='pp_semantics':
            structure='pp'
            feature='animacy'
            title=r'$\mathcal{PP-Animacy}$'
            
        # parse the data
        q=f"structure=='{structure}' & feature=='{feature}'"
        df=data.query(q)
  

  
    
        print(2*'\n',20*'*',f'{construction}', 20*'*')
        print(20*'+','VIOLATION EFFECT', 20*'+',)
        mean_violation=round(df[df.violation==1].error_per_combination.mean(),2)
        mean_grammatical=round(df[df.violation==0].error_per_combination.mean(),2)
        std_violation=round(df[df.violation==1].error_per_combination.sem(),2)
        std_grammatical=round(df[df.violation==0].error_per_combination.sem(),2)

        print(f'Grammatical: {mean_grammatical}+-{std_grammatical}')
        print(f'Violation: {mean_violation}+-{std_violation}')


        print(20*'+','CONGRUENCY EFFECT', 20*'+',)
        mean_CONGRUENT=round(df[df.congruency==1].error_per_combination.mean(),2)
        mean_INCONGRUENT=round(df[df.congruency==0].error_per_combination.mean(),2)
        std_CONGRUENT=round(df[df.congruency==1].error_per_combination.sem(),2)
        std_INCONGRUENT=round(df[df.congruency==0].error_per_combination.sem(),2)

        print(f'Grammatical: {mean_CONGRUENT}+-{std_CONGRUENT}')
        print(f'Violation: {mean_INCONGRUENT}+-{std_INCONGRUENT}')


get_summary_statistics(data)            