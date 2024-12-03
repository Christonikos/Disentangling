#!/usr/bin/env python
from utils import data, lstm
import numpy as np
import pickle
from tqdm import tqdm
import sys
import math
import os
import torch
import argparse
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'word_language_model/')))


parser = argparse.ArgumentParser(
    description='Extract model activation during sentence processing.')
parser.add_argument('--model', type=str,
                    default='../../models/hidden650_batch128_dropout0.2_lr20.0.pt',
                    help='Meta file stored once finished training the corpus')
parser.add_argument('--stimuli', default='../../data/stimuli/stimuli.csv',
                    help='Input sentences')
parser.add_argument('-v', '--vocabulary', default='../../models/vocab.txt')
parser.add_argument('--path2output',
                    default='../../data/activations/',
                    help='Destination for the output vectors')
parser.add_argument('--perplexity', action='store_true', default=False)
parser.add_argument('--eos-separator', default='</s>')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--use-unk', action='store_true', default=True)
parser.add_argument('--lang', default='en')
parser.add_argument('--unk-token', default='<unk>')
parser.add_argument('--uppercase-first-word',
                    action='store_true', default=False)

args = parser.parse_args()
df_meta = pd.read_csv(args.stimuli)
sentences = df_meta['sentence'].to_numpy()
sentences = [s.strip('\n').strip('.').split() for s in sentences]

print(args.model)
if not args.cuda:
    model = torch.load(args.model, map_location='cpu')
else:
    model = torch.load(args.model)

# Hack the forward function to send an extra argument with model parameters
model.rnn.forward = lambda input, hidden: lstm.forward(model.rnn, input, hidden)
vocab = data.Dictionary(args.vocabulary)
saved = {}


def feed_sentence(model, h, sentence):
    outs = []
    for w in sentence:
        out, h = feed_input(model, h, w)
        outs.append(torch.nn.functional.log_softmax(out[0]).unsqueeze(0))
    return outs, h


def feed_input(model, hidden, w):
    if w not in vocab.word2idx and args.use_unk:
        print('unk word: ' + w)
        w = args.unk_token
    inp = torch.autograd.Variable(torch.LongTensor([[vocab.word2idx[w]]]))
    if args.cuda:
        inp = inp.cuda()
    out, hidden = model(inp, hidden)
    return out, hidden


print('Extracting LSTM representations', file=sys.stderr)
# output buffers
log_probabilities = [np.zeros(len(s)) for s in sentences]
vectors = {k: [np.zeros((model.nhid*model.nlayers, len(s))) for s in sentences]
           for k in ['gates.in', 'gates.forget', 'gates.out',
                          'gates.c_tilde', 'hidden', 'cell']}
if args.lang == 'en':
    init_sentence = " ".join(["In service , the aircraft was operated by a crew of five and could accommodate either 30 paratroopers , 32 <unk> and 28 sitting casualties , or 50 fully equipped troops . <eos>",
                    "He even speculated that technical classes might some day be held \" for the better training of workmen in their several crafts and industries . <eos>",
                    "After the War of the Holy League in 1537 against the Ottoman Empire , a truce between Venice and the Ottomans was created in 1539 . <eos>",
                    "Moore says : \" Tony and I had a good <unk> and off-screen relationship , we are two very different people , but we did share a sense of humour \" . <eos>",
                    "<unk> is also the basis for online games sold through licensed lotteries . <eos>"])
    # init_sentence = "</s>"
hidden = model.init_hidden(1)
init_sentence = [s.lower() for s in init_sentence.split(" ")]
init_out, init_h = feed_sentence(model, hidden, init_sentence)
for i, words in enumerate(tqdm(sentences)):
    out = init_out[-1]
    hidden = init_h
    for j, w in enumerate(words):
        if j == 0 and args.uppercase_first_word:
            w = w.capitalize()

        if w not in vocab.word2idx and args.use_unk:
            print('unk word: ' + w)
            w = args.unk_token
        # store the surprisal for the current word
        log_probabilities[i][j] = out[0, 0, vocab.word2idx[w]].data.item()
        inp = torch.autograd.Variable(torch.LongTensor([[vocab.word2idx[w]]]))
        if args.cuda:
            inp = inp.cuda()
        out, hidden = model(inp, hidden)
        out = torch.nn.functional.log_softmax(out[0]).unsqueeze(0)

        if not args.perplexity:
            vectors['hidden'][i][:, j] = \
                hidden[0].data.view(1, 1, -1).cpu().numpy()
            vectors['cell'][i][:, j] = \
                hidden[1].data.view(1, 1, -1).cpu().numpy()
            # we can retrieve the gates thanks to the hacked function
            for k, gates_k in vectors.items():
                if 'gates' in k:
                    k = k.split('.')[1]
                    gates_k[i][:, j] = \
                        torch.cat([g[k].data for g
                                   in model.rnn.last_gates], 1).cpu().numpy()
    # save the results
    saved['log_probabilities'] = log_probabilities
    saved['sentences'] = sentences
    
    if not args.perplexity:
        for k, g in vectors.items():
            saved[k] = g

print ("Perplexity: {:.2f}".format(
        math.exp(
                sum(-lp.sum() for lp in log_probabilities)/
                sum((lp!=0).sum() for lp in log_probabilities))))


fn = f'{os.path.basename(args.model)}_stimuli.pkl'
with open(os.path.join(args.path2output, fn), 'wb') as fout:
    pickle.dump(saved, fout, -1)
