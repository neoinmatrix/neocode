import os

from matplotlib import pylab
import numpy as np

def plot_confusion_matrix(cm, genre_list, name, title,max,save=False):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Greens', vmin=0, vmax=max)
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    pylab.title(title)
    pylab.colorbar()
    # pylab.grid(True)
    pylab.show()
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(True)
    if save==True:
        pylab.savefig(os.path.join(CHART_DIR, "confusion_matrix_%s.png"%name), bbox_inches="tight")



def plot_confusion_matrix2(cm, genre_list, name, title,save=False):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Greens', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(np.arange(0,len(genre_list),1))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(np.arange(0,len(genre_list),1))
    ax.set_yticklabels(genre_list)
    pylab.title(title)
    pylab.colorbar()
    # pylab.grid(True)
    pylab.show()
    pylab.xlabel('x class')
    pylab.ylabel('y class')
    pylab.grid(True)
    if save==True:
        pylab.savefig(os.path.join(CHART_DIR, "confusion_matrix_%s.png"%name), bbox_inches="tight")


