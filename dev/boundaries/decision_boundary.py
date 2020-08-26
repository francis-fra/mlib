import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

def plot_decision_contour(clf, X, y, show_contour=True, newPlot=False):
    "plot decision contour"
    
    if newPlot:
        fig, ax = plt.subplots()
    else:
        ax = plt.gca()        
        
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    h = .02  # step size in the mesh
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            
            
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    if show_contour:
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
    # change to one dimension
    X_new = np.c_[xx.ravel(), yy.ravel()]
    y_pred = clf.predict(X_new).reshape(xx.shape)
    # decision boundary
    im = plt.contour(xx, yy, y_pred, alpha=0.8, colors='k', linewidths=0.8, linestyles='-')

    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
            

def plot_decision_boundary(clf, X, y, N=100, alpha=0.3, bg=True, newPlot=False):
    "plot decision boundary"
    
    if newPlot:
        fig, ax = plt.subplots()
    else:
        ax = plt.gca()
    
    # find plot limits
    xmin, xmax = min(X[:,0]), max(X[:,0])
    ymin, ymax = min(X[:,1]), max(X[:,1])
    
    # extend range
    xmin = xmin*1.1 if xmin < 0 else xmin*0.9
    xmax = xmax*1.1
    ymin = ymin*1.1 if ymin < 0 else ymin*0.9
    ymax = ymax*1.1
    
    x1s = np.linspace(xmin, xmax, N)
    x2s = np.linspace(ymin, ymax, N)
    
    # number of class
    classes = list(set(y))
    numClass = len(classes)
    
    # mesh grid
    x1, x2 = np.meshgrid(x1s, x2s)
    
    # change to one dimension
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    
    # TODO: change color map
    styles = ["b^", "rs", "y^", "m*", "c1", "g2", "k3"]
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    
    if bg == True:
        plt.contourf(x1, x2, y_pred, alpha=alpha, cmap=custom_cmap)

    # contour
    custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
    im = plt.contour(x1, x2, y_pred, alpha=alpha, colors='k', linewidths=0.5, linestyles='-')

    # scatter plot
    for k in range(numClass):
        style = styles[k]
        a = classes[k]
        plt.plot(X[:, 0][y==a], X[:, 1][y==a], style)
        
    return im

def plot_regression_predictions(reg, X, y, N=500, transformer=None):
    "regression line"
    
    # find plot limits
    xmin, xmax = min(X), max(X)
    ymin, ymax = min(y), max(y)
    
    x1 = np.linspace(xmin, xmax, N).reshape(-1, 1)
    if transformer is not None:
        x_train = transformer(x1)
    else:
        x_train = x1
        
    y_pred = reg.predict(x_train)
    
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2)
    plt.grid()
    
    
def animate_boundary(X, y, clf=None, max_estimators=200, numFrames=30, repeat=False):
    "animate boundary changes"
    
    frameList = np.linspace(1, max_estimators, numFrames, dtype=np.int32)

    def update_baggingClassifier(k, X, y, ax, fig):
        ax.cla()
        bag_clf = BaggingClassifier(
            clf, n_estimators=k, max_samples=1.0, bootstrap=True, n_jobs=-1)
    
        bag_clf.fit(X, y)
        #im = plot_decision_boundary(bag_clf, X, y, newPlot=False)
        im = plot_decision_contour(bag_clf, X, y, newPlot=False)
        plt.title('Bagging: {} Estimators'.format(k))
        return im,

    def update_randomForest(k, X, y, ax, fig):
        ax.cla()
        rnd_clf = RandomForestClassifier(n_estimators=k, n_jobs=-1)   
        rnd_clf.fit(X, y)
        #im = plot_decision_boundary(rnd_clf, X, y, newPlot=False)
        im = plot_decision_contour(rnd_clf, X, y, newPlot=False)
        plt.title('Random Forest: {} Trees '.format(k))
        return im,

    # num animate frames
    def animate(clf):
        fig = plt.figure()
        ax = fig.gca()
        
        if clf is None:
            update_func = update_randomForest
        else:
            update_func = update_baggingClassifier
             
        ani = animation.FuncAnimation(fig, update_func, 
                                      frames=frameList, 
                                      fargs=(X, y, ax, fig), 
                                      interval=1,
                                      repeat = repeat)

        return ani

    ani = animate(clf)
    return ani

    