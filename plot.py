from explore import is_column_categorical
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utility import get_categorical_interval

#============================================================
# plotting
#============================================================
# def one_way_analysis(df, x, y, numbins=10):
#     "histogram of x and mean y wrt to x"
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax2 = plt.twinx()
#     ax1.set_ylabel(y + ' %')
#     ax1.set_xlabel(x)
#     # right 
#     results, bins = pd.cut(df[x], numbins, retbins=True)
#     bin_hist = results.value_counts().sort_index()
#     ax2.bar(range(numbins), bin_hist.values, zorder=1)
#     ax2.set_ylabel('Profile Count')
#     # left: line plot
#     ax1.set_zorder(ax2.get_zorder()+1)
#     tmp = df.groupby(results)[y].mean()
#     ax1.plot(range(numbins), tmp.values, 'ro-')
#     ax1.patch.set_visible(False) 



def one_way_analysis(sdf, x, y, model=None, numbins=10):
    """
        plot 1) histogram of x and 2) mean y wrt to binned x

        Parameters
        ----------
        sdf    : data frame
        x      : string
        y      : binary target column name in string
        model  : statmodel
        numbins : num bins
    """
    df = sdf.copy()
    if model is not None:
        df['pred'] = model.predict(df)

    # check if binning is required
    if not is_column_categorical(df, x):
        dfx, bins = pd.cut(df[x], numbins, retbins=True)
        # mean of y conditional on x
        avg_line = df.groupby(dfx)[y].mean()
        # avg_line = (bins[:-1] + bins[1:]) / 2
        if model is not None:
            pred_line = df.groupby(dfx)['pred'].mean()
        # xx = get_categorical_interval(avg_line.index)
        xx = (bins[:-1] + bins[1:]) / 2
    else:
        cnts = df[x].value_counts().sort_index()
        avg_line = df.groupby(df[x])[y].mean()
        if model is not None:
            pred_line = df.groupby(dfx)['pred'].mean()
        xx = cnts.index
        # no binning - reset numbins to natural unique values
        numbins = len(xx)

    # plotting
    _, ax1 = plt.subplots()
    # avg line plot
    sns.lineplot(x=xx, y=avg_line.values, marker='o', color='crimson', ax=ax1)
    if model is not None:
        sns.lineplot(x=xx, y=pred_line.values, marker='o', color='black', ax=ax1)
    ax1.legend(labels=["actual","prediction"])
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.set_ylabel(y + ' %')
    # histogram
    sns.histplot(df[x], bins=numbins, ax=ax2)
    ax2.set_ylabel('Profile Count')
    ax1.patch.set_visible(False)

def single_profile_analysis(sdf, col, model, numbins=20):
    """
        step 1: select a row of data 
        step 2: select a feature variable (v) and bin it
        repeat for each bin value:
        step 3: replace the actual v with different v
        step 4: predict using model and plot the response rate
    """
    dfx, _ = pd.cut(sdf[col], numbins, retbins=True)
    avg = sdf[col].groupby(dfx).mean()
    # TODO: set as average (or mode) for other variables as an option
    row = sdf.sample(n=1)
    df = pd.concat([row]*numbins, axis=0).reset_index(drop=True)
    for idx, v in enumerate(avg):
        df.loc[idx, col] = v
    pred = model.predict(df)
    _ ,ax1 = plt.subplots()
    sns.lineplot(x=avg.values, y=pred, marker='o',  color='crimson', ax = ax1)
    ax1.set_ylabel('mean prediction')
    ax1.set_xlabel(col)
    ax1.grid()

# Average Curve Analysis
def avg_curve_analysis(df, colname, model, num_bins=10, with_const=True):
    """
        Parameters
        ----------
        df    : data frame
        colname     : name of independent variable
        model : statmodel
        with_const : if there is a const coluum
    """
    result = []
    # feature_list = list(model.params.index)
    feature_list = df.columns
    dfx, bins = pd.cut(df[colname], num_bins, retbins=True)
    values = (bins[:-1] + bins[1:]) / 2
    rdf = df.copy()
    for v in values:
        if with_const:
            rdf['const'] = 1.0
        # set the whole column to be the same value
        rdf[colname] = v
        X = rdf[feature_list]
        result.append(model.predict(X).mean())
    sns.lineplot(x=values, y=result, marker='o',  color='crimson')
    plt.ylabel("avg prediction")
    plt.xlabel(colname)
    plt.grid()

def fit_analysis(sdf, target_col, model):
    """
        actual vs predicted

        Parameters
        ----------
        sdf    : data frame
        target_col: string
        model : statmodel
    """

    df = sdf.copy()
    numbins = 10
    _ ,ax1 = plt.subplots()
    ax1.set_ylabel('actual mean probability')
    ax1.set_xlabel('predicted probability')
    df['pred']= model.predict(df)
    actual = df[target_col]
    # equal space binned predicted prob
    dfx, _ = pd.cut(df['pred'], numbins, retbins=True)
    # actual mean
    # values = (bins[:-1] + bins[1:]) / 2
    avg = actual.groupby(dfx).mean()
    xx = get_categorical_interval(avg.index)
    sns.lineplot(x=xx, y=avg.values, marker='o', linestyle='', color='crimson', ax=ax1)
    sns.lineplot(x=xx, y=xx, linestyle='--', color='black', ax=ax1)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Prediction Count')
    sns.histplot(df['pred'], bins=numbins, ax=ax2)

# TORM
# def one_way_analysis(df, x, y, numbins=10):
#     "histogram of x and mean y wrt to binned x"
#     # check if binning is required
#     if not is_column_categorical(df, x):
#         dfx, _ = pd.cut(df[x], numbins, retbins=True)
#         avg_line = df.groupby(dfx)[y].mean()
#         xx = get_categorical_interval(avg_line.index)
#     else:
#         cnts = df[x].value_counts().sort_index()
#         avg_line = df.groupby(df[x])[y].mean()
#         xx = cnts.index
#         # no binning - reset numbins to natural unique values
#         numbins = len(xx)

#     # plotting
#     fig, ax1 = plt.subplots()
#     # avg line plot
#     sns.lineplot(x=xx, y=avg_line.values, marker='o', color='crimson', ax=ax1)
#     ax2 = ax1.twinx()
#     ax1.set_zorder(ax2.get_zorder()+1)
#     ax1.set_ylabel(y + ' %')
#     # histogram
#     sns.histplot(df[x], bins=numbins, ax=ax2)
#     ax2.set_ylabel('Profile Count')
#     ax1.patch.set_visible(False)


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
