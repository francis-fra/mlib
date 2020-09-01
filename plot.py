import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#============================================================
# plotting
#============================================================
def plot_top_counts(data, ishor=False):
    keys = [item[0] for item in data]
    values = [item[1] for item in data]
    if ishor == False:
        df = pd.DataFrame({'categories': keys, 'counts': values})
        ax = df.plot.bar(x='categories', y='counts', rot=0)
    else:
        keys.reverse()
        values.reverse()
        df = pd.DataFrame({'categories': keys, 'counts': values})
        ax = df.plot.barh(x='categories', y='counts', rot=0)

def plot_series(s, ishor=False):
    "bar chart for series"
    if ishor == True:
        s = s.sort_values(ascending=True)
        df = pd.DataFrame(s).plot.barh()
    else:
        df = pd.DataFrame(s).plot.bar()

def plot_multiclass_roc_curve(plotData):
    
    for idx in plotData['label_map'].keys():
        plt.plot(plotData['fpr'][idx], plotData['tpr'][idx], \
                    label=plotData['label_map'][idx])
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()

def plot_roc_curve(plotData):
    """
        dictionary containing at most two curve data
        e.g.
        plotdata = {
            'fpr': fpr,
            'tpr': tpr,
            'primary': 'test',
            'fitted_fpr': fitted_fpr,
            'fitted_tpr': fitted_tpr,
            'secondary': 'fitted'
        }
        fitted lines can be None
    """
    # random line
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='Random')

    plt.plot(plotData['fpr'], plotData['tpr'], 'r--', label=plotData['primary'])
    if plotData['fitted_fpr'] is not None and plotData['fitted_tpr'] is not None:
        plt.plot(plotData['fitted_fpr'], plotData['fitted_tpr'], 'b--', label=plotData['secondary'])

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()

def plot_confusion_matrix(confmat, labels):
    "plot confusion matrix"

    fig, ax = plt.subplots()
    caxes = ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.8)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.tight_layout()

def plot_drivers(df):
    """plot drivers

        Parameters:
        -----------
        df : df with two columns: name and value
    
    """

    left = 0.25
    bottom = 0.15
    width = 0.7
    height = 0.8
    size = (8, 6)
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([left, bottom, width, height])

    y_pos = np.arange(len(df.name))
    ax.barh(y_pos, df.value)
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.name)
    plt.title("Key Drivers")
    plt.xlabel('value')
