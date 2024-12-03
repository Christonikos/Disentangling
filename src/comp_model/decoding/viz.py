import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)

def get_color_str(word, timepoint):
    if word in ['n', 'v']:
        curr_str = f'{word.upper()}'
        if timepoint in [3, 18]:
            color = 'r'
            curr_str += '1'
        elif timepoint in [6, 15]:
            color = 'g'
            curr_str += '2'
        elif timepoint in [9, 12]:
            color = 'b'
            curr_str += '3'
    else:
        curr_str = ''
        color = 'k'
    return color, curr_str


def plot_GAT(scores, verb_position, words, title):
    num_words = len(words)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # GAT
    im = axs[0].matshow(scores, vmin=0, vmax=1., cmap='RdBu_r',
                        origin='lower', extent=[0, num_words, 0, num_words])
    axs[0].xaxis.set_ticks_position('bottom')
    axs[0].set_xlabel('Testing Time (s)')
    axs[0].set_ylabel('Training Time (s)')
    axs[0].set_xticks(np.arange(0.5, num_words + 0.5))
    axs[0].set_xticklabels(words)
    axs[0].set_yticks(np.arange(0.5, num_words + 0.5))
    axs[0].set_yticklabels(words)
    cbar = plt.colorbar(im, ax=axs[0])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('AUC', rotation=270)
    for t in range(0, num_words, 1):
        axs[0].axvline(t, color='k', ls='--')
        axs[0].axhline(t, color='k', ls='--')
    # DIAG
    axs[1].plot(range(1, num_words+1), np.diag(scores), label='diag')
    # axs[1].plot(range(1, num_words + 1), scores[verb_position, :])
    axs[1].axhline(.5, color='k', linestyle='--', label='chance')
    # axs[1].axvline(verb_position, color='g', linestyle='--')
    # axs[1].axvline(num_words - verb_position, color='g', linestyle='--')
    axs[1].set_xlim([1, num_words+1])
    axs[1].set_ylim([-0.1, 1.1])
    axs[1].set_xlabel('Times')
    axs[1].set_ylabel('AUC')
    axs[1].set_xticks(range(1, len(words)+1))
    axs[1].set_xticklabels(words)
    axs[1].legend(loc=4)
    axs[1].axvline(.0, color='k', ls='-')

    fig.suptitle(title)

    return fig, axs