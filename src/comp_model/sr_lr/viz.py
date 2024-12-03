#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 23:22:18 2021

@author: yl254115
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_sr_lr_competition(contribution_to_prediction, df_meta,
                           structure, viol):
    
    unit_types = ['LR', 'SR']
    line_colors = ['g', 'r']
    line_widths = [3, 3]
    line_styles = ['-', '-']
    
    d = 2
    number_values= ['singular', 'plural']
    conditions = [(n1, n2) for n1 in number_values for n2 in number_values]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for c, cond in enumerate(conditions):
        query = f"feature=='number' & main_noun_number=='{cond[0]}' & embedded_noun_number=='{cond[1]}' & structure=='{structure}' & violIndex=={viol}"
        IXs = df_meta.query(query).index.to_numpy()
        xlabels = df_meta['sentence'][IXs[0]].rstrip('\n').rstrip('.').split()
        for ut, unit_type in enumerate(unit_types):
            label = {'LR': 'Global', 'SR':'Local'}[unit_type]
            num_samples = contribution_to_prediction[unit_type][IXs].shape[0]
            cond_ave = np.nanmean(contribution_to_prediction[unit_type][IXs], axis=0)
            cond_std = np.nanmean(contribution_to_prediction[unit_type][IXs], axis=0)
            axs[c % d, int(np.floor(c / d))].errorbar(range(len(cond_ave)),
                                                      cond_ave,
                                                      yerr=cond_std/np.sqrt(num_samples),
                                                      label=label,
                                                      lw=line_widths[ut],
                                                      color=line_colors[ut],
                                                      ls=line_styles[ut])
            axs[c % d, int(np.floor(c / d))].set_xticks(range(len(xlabels)))
            axs[c % d, int(np.floor(c / d))].set_xticklabels(xlabels, fontsize=12)
            title = '-'.join(cond)
            axs[c % d, int(np.floor(c / d))].set_title(title, fontsize=14)
            axs[c % d, int(np.floor(c / d))].set_ylim([-1, 1])
            axs[c % d, int(np.floor(c / d))].axhline(0, color='k', ls='--', lw=1)
            if c < d:
                axs[c % d, int(np.floor(c / d))].set_ylabel('$\\alpha_{correct/wrong}$', fontsize=30)

    plt.legend(fontsize=18, bbox_to_anchor=(0.3, -0.1), loc='bottom', ncol=2)
    return fig, axs
