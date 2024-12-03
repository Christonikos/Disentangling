import pickle, sys, os
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import cross_val_score
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src/word_language_model')))
import data


def parseNumList(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise argparse.ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start,10), int(end,10)))



parser = argparse.ArgumentParser(description='Find short-range (SR) units')
parser.add_argument('--model', type=str, default='../../../Data/LSTM/models/Italian_hidden650_batch64_dropout0.2_lr20.0.pt', help='pytorch model')
parser.add_argument('-v', '--verbs', default='../../../Data/Stimuli/singular_plural_verbs_Italian.txt',
					help='Text file with two tab delimited columns with the lists of output words to contrast with the PCA')
parser.add_argument('-u', '--units', default='650-1300', nargs='+', type=parseNumList, help='Which units in 2ND LAYER to analyze (counting from ZERO! The second number in the default 650-1300 means upto but not including 1300 as in range().)')
parser.add_argument('--vocabulary', default='../../../Data/LSTM/models/Italian_vocab.txt')
parser.add_argument('-s', '--sentences', type=str, default='../../../Data/Stimuli/Italian_simple_non_512.text', help='Path to text file containing the list of sentences to analyze')
parser.add_argument('-m', '--metadata', type=str, default='../../../Data/Stimuli/Italian_simple_non_512.info', help='The corresponding meta data of the sentences')
parser.add_argument('-a', '--activations', type=str, default='../../../Data/LSTM/activations/Italian/simple_non.pkl', help='The corresponding sentence (LSTM) activations')
parser.add_argument('-g', '--gate', type=str, default='cell', help='One of: gates.in, gates.forget, gates.out, gates.c_tilde, hidden, cell')
parser.add_argument('-t', '--timepoints', default=[1, -1], help='Two time points for subject (1st value) and attractor (2nd) on which classification is and GAT are tested (counting from ZERO!). Default points are for nounpp')
parser.add_argument('--threshold', default=0.9, help='Threshold for decoding performance. Shared for all type of tests (0.92 is the minimal thresh to have the plural unit 775 detected)')
parser.add_argument('-d', '--d-layer', default=650, help='Assuming only 2 layers of equal size (!), the dimension of the second layer (#units)')
parser.add_argument('-o', '--output-file-name', type=str, default='../../../Output/SR_LR_units.txt', help='Path to output folder for figures')
parser.add_argument('--test-activations', action='store_false', default=True)
parser.add_argument('--test-embeddings', action='store_false', default=True)
args = parser.parse_args()
print(args)

# LR_units = [769, 987, 775]
LR_units = [814]

def get_labels(metadata, condition_list_class1, condition_list_class2):
    labels = np.zeros(len(metadata))
    for i, curr_info in enumerate(metadata):

        is_in_class1 = all(curr_info[f] == v for (f,v) in condition_list_class1)
        if is_in_class1:
            labels[i] = 1

        is_in_class2 = all(curr_info[f] == v for (f, v) in condition_list_class2)
        if is_in_class2:
            labels[i] = 2

    return labels


def get_decoding_scores(X, y):
    clf = svm.SVC(kernel='linear', C=1)
    return cross_val_score(clf, X, y, cv=5, n_jobs=-1)


def get_decoding_GAT(X1, y1, X2):
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X1, y1)
    return clf.score(X2, y1)

# -----------------
# --- Load data ---
# -----------------
metadata = pickle.load(open(args.metadata, 'rb'))
activations = pickle.load(open(args.activations, 'rb'))
activations = np.dstack(activations[args.gate])
activations = np.moveaxis(activations, 2, 0) # num_trials X num_units X num_timepoints
with open(args.sentences, 'r') as f:
    sentences = f.readlines()
sentences = [s.split(' ') for s in sentences]

# ---------------------------------------
# --- Decode feature from activations ---
# ---------------------------------------
number_units_based_on_activations_decoding = args.units
if args.test_activations:
    scores_from_activations_per_unit = {} # list of scores from all random CV seeds
    for u, unit in tqdm(enumerate(args.units)):
        # print(u)

        t_subject   = args.timepoints[0]
        t_attractor = args.timepoints[1]

        # conditions SP and PS; class-'1': singular, class-'2': plural
        data_X_subject = np.squeeze(activations[:, unit, t_subject]).reshape(-1, 1)
        if args.timepoints[1] > -1:
            data_X_attractor = np.squeeze(activations[:, unit, t_attractor]).reshape(-1, 1)

        if args.timepoints[1] > -1:
            data_y_subject = get_labels(metadata, [('number_1', 'singular'), ('number_2', 'plural')], [('number_1', 'plural'), ('number_2', 'singular')])
        else:
            data_y_subject = get_labels(metadata, [('number_1', 'singular')], [('number_1', 'plural')])
        IX_classes = np.where(data_y_subject > 0)  # '0' in data_y if sentence is in neither class ('1' or '2' elsewhere)
        data_y_subject = data_y_subject[IX_classes]
        if args.timepoints[1] > -1:
            data_y_attractor = 3 - data_y_subject

        data_X_subject = data_X_subject[IX_classes]
        if args.timepoints[1] > -1:
            data_X_attractor = data_X_attractor[IX_classes]

        scores_on_subject = get_decoding_scores(data_X_subject, data_y_subject)
        if args.timepoints[1] > -1:
            scores_on_attractor = get_decoding_scores(data_X_attractor, data_y_attractor)
            scores_GAT_subject_to_attractor = get_decoding_GAT(data_X_subject, data_y_subject, data_X_attractor)

        if args.timepoints[1] > -1:
            scores_from_activations_per_unit[unit] = (np.mean(scores_on_subject), np.mean(scores_on_attractor), scores_GAT_subject_to_attractor)
        else:
            scores_from_activations_per_unit[unit] = (np.mean(scores_on_subject))

    if args.timepoints[1] > -1:
        SR_units_based_on_activations_decoding = [unit for (unit, scores) in scores_from_activations_per_unit.items() if
                                                  all([scores[0]>=args.threshold, scores[1]>=args.threshold,
                                                       scores[2]<1-args.threshold])]
        LR_units_based_on_activations_decoding = [unit for (unit, scores) in scores_from_activations_per_unit.items() if
                                                      all([scores[0] >= args.threshold, scores[1] >= args.threshold,
                                                           scores[2] > args.threshold])]
    else:
        SR_units_based_on_activations_decoding = [unit for (unit, scores) in scores_from_activations_per_unit.items() if
                                                  scores >= args.threshold]
        LR_units_based_on_activations_decoding = [unit for (unit, scores) in scores_from_activations_per_unit.items() if
                                                  scores >= args.threshold]

# -------------------------------------------
# --- Decode feature from decoder weights ---
# -------------------------------------------
number_units_based_on_embeddings_decoding = args.units
if args.test_embeddings:
    print('Loading model...')
    model = torch.load(args.model, lambda storage, loc: storage)
    model.rnn.flatten_parameters()
    embeddings_in = model.encoder.weight.data.cpu().numpy()
    embeddings_out = model.decoder.weight.data.cpu().numpy()
    vocab = data.Dictionary(args.vocabulary)

    # Read list of contrasted words (e.g., singular vs. plural verbs).
    with open(args.verbs, 'r') as f:
        lines=f.readlines()
    verbs_singular = [l.split('\t')[0].strip() for l in lines]
    verbs_plural = [l.split('\t')[1].strip() for l in lines]
    verbs_all = verbs_singular + verbs_plural
    print('\nWords used (group 1):')
    print(verbs_singular)
    print('\nWords used (group 2):')
    print(verbs_plural)

    # Get index in the vocab for all words and extract embeddings
    idx_verbs_singular = [vocab.word2idx[w] for w in verbs_singular]
    idx_verbs_plural = [vocab.word2idx[w] for w in verbs_plural]
    idx_verbs_all = idx_verbs_singular + idx_verbs_plural
    embeddings_verbs_out_singular = embeddings_out[idx_verbs_singular, :]
    embeddings_verbs_out_plural = embeddings_out[idx_verbs_plural, :]
    embeddings_verbs_out_all = embeddings_out[idx_verbs_singular + idx_verbs_plural ,:] # num_verbs X num_units_last_layer
    data_y = '1'*len(idx_verbs_singular) + '2' * len(idx_verbs_plural)
    data_y = np.asarray([int(j) for j in data_y])

    scores_from_embeddings_per_unit = {} # list of scores from all random CV seeds
    for u, unit in tqdm(enumerate(args.units)):
        data_X = np.squeeze(embeddings_verbs_out_all[:, unit-args.d_layer]).reshape(-1, 1)
        curr_scores = get_decoding_scores(data_X, data_y)
        scores_from_embeddings_per_unit[unit] = np.mean(curr_scores)

    number_units_based_on_embeddings_decoding = [unit for (unit, score) in scores_from_embeddings_per_unit.items() if score>=args.threshold]

    with open('AUC_all_units.pkl', 'wb') as f:
        pickle.dump(scores_from_embeddings_per_unit, f)
    import matplotlib.pyplot as plt
    fig = plt.subplots(figsize=(10, 10))
    plt.hist(scores_from_embeddings_per_unit.values(), 100)
    plt.show()
# ---------------------
# --- Write to file ---
# ---------------------
SR_units = list(set(number_units_based_on_embeddings_decoding)-set(LR_units))
with open(args.output_file_name, 'w') as f:
    f.writelines(', '.join([str(u) for u in SR_units])+'\n')
    f.writelines(', '.join([str(u) for u in LR_units])+'\n')
    f.writelines(', '.join([str(k) + ':' + str(v) for (k, v) in vars(args).items()]))


verbose = True
if verbose:
    SR_units = list(set(number_units_based_on_embeddings_decoding) & set(SR_units_based_on_activations_decoding))
    LR_units = list(set(number_units_based_on_embeddings_decoding) & set(LR_units_based_on_activations_decoding))

    print('\n\nTest 1 - SR activation decoding:\n', SR_units_based_on_activations_decoding)
    print('\n\nTest 1 - LR activation decoding:\n', LR_units_based_on_activations_decoding)
    print('\n\nTest 2 - embeddings decoding:\n', number_units_based_on_embeddings_decoding)
    print('\n\nSR units:\n', SR_units)

    with open(args.output_file_name, 'w') as f:
        f.writelines('Test 1 - SR unit numbers that pass decoding performance greater than %1.3f based on unit activations:\n' % args.threshold)
        f.writelines(', '.join([str(u) for u in SR_units_based_on_activations_decoding])+'\n')
        f.writelines('Test 1 - LR unit numbers that pass decoding performance greater than %1.3f based on unit activations:\n' % args.threshold)
        f.writelines(', '.join([str(u) for u in LR_units_based_on_activations_decoding]) + '\n')
        f.writelines('Test 2 - unit numbers that pass decoding performance greater than %1.3f based on embeddings:\n' % args.threshold)
        f.writelines(', '.join([str(u) for u in number_units_based_on_embeddings_decoding])+'\n')
        f.writelines('SR units - unit numbers that pass all tests:\n')
        f.writelines(', '.join([str(u) for u in SR_units]))
        f.writelines('LR units - unit numbers that pass all tests:\n')
        f.writelines(', '.join([str(u) for u in LR_units]))