import pickle
import sys
import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from utils import data, lstm
from sr_lr.utils import get_output_activations, get_sr_lr_predictions
from sr_lr.viz import plot_sr_lr_competition
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname
                                             (os.path.realpath(__file__)),
                                             'word_language_model')))

parser = argparse.ArgumentParser(
    description='Decompose prediction to SR and LR contributions')
parser.add_argument('--model', type=str, default=
                    '../../models/hidden650_batch128_dropout0.2_lr20.0.pt',
                    help='pytorch model')
parser.add_argument('--path2stimuli', default='../../data/stimuli/stimuli.csv')
parser.add_argument('--path2activations', default='../../data/activations/hidden650_batch128_dropout0.2_lr20.0.pt_stimuli.csv.pkl')
parser.add_argument('--vocabulary', default='../../models/vocab.txt')
parser.add_argument('--lr-units',
                    default=[987, 775], nargs='+', #769, 775, 987
                    help='Long-range unit numbers - counting from ZERO!')
parser.add_argument('--sr-units',
                    default=[1282, 1283, 772, 905, 1035, 1167, 1295, 655,
                             1042, 1171, 916, 661, 1052, 796, 925, 671, 1055,
                             1058, 1065, 681, 682, 684, 939, 1199, 1202, 1203,
                             950, 952, 1210, 699, 1214, 702, 831, 833, 714,
                             972, 847, 975, 978, 1235, 851, 853, 856, 857,
                             1115, 745, 1006, 1264, 884], nargs='+',
                    help='Short-range unit numbers - counting from ZERO!')
parser.add_argument('--path2output', type=str,
                    default='../../results/sr_lr/', help='Path to output')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--use-unk', action='store_true', default=True)
parser.add_argument('--unk-token', default='<unk>')
args = parser.parse_args()
print(args)


# LOAD
print('Loading sentence stimuli and their metadata')
df_meta = pd.read_csv(args.path2stimuli)
sentences = df_meta['sentence'].to_numpy()
vocab = data.Dictionary(args.vocabulary)

model = torch.load(args.model, lambda storage, loc: storage)
model.rnn.forward = \
    lambda input, hidden: lstm.forward(model.rnn, input, hidden)

# CALC ACTIVATIONS AT OUTPUT LAYER
output_activations = get_output_activations(model, args.path2activations,
                                            args.lr_units, args.sr_units)

# CALC RELATIVE CONTRIBUTION OF SR AND LR UNITS
contribution_to_prediction = get_sr_lr_predictions(output_activations,
                                                   df_meta, vocab)

# PLOT
for viol in range(2):
    for structure in ['pp', 'obj']:
        fig, axs = plot_sr_lr_competition(contribution_to_prediction,
                                          df_meta,
                                          structure, viol)
        fname = f'SR_LR_interplay_{structure}_viol_{viol}.png'
        fname = os.path.join('..', '..', 'figures', 'sr_lr', fname)
        plt.savefig(fname)
        print('Figure saved to: ', fname)
