import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd



def biplot(x, y, xlabel='X', ylabel='Y', title='Biplot',
           l1=None, l2=None):
    '''
    x and y - numppy.ndarray with two columns
    '''

    f = plt.figure(figsize=(12, 8))
    ax = f.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(x[:, 0], x[:, 1], c='r', label='Set X')
    ax.scatter(y[:, 0], y[:, 1], c='b', label='Set Y')
    if l1 is not None:
        for i in range(len(l1)):
            ax.text(x[i, 0], x[i, 1], l1[i])
    if l2 is not None:
        for i in range(len(l2)):
            ax.text(y[i, 0], y[i, 1], l2[i])
    ax.legend()


# building the correlogram based on the correlation matrix
def correlogram(R2, dec=2, title='Correlogram', valmin=-1, valmax=1):
    plt.figure(title, figsize=(8, 6))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(R2, dec), vmin=valmin, vmax=valmax,
               cmap='bwr', annot=True)


# creates the graphical table based on the intensity of link between
# values on X and Y axes
def intensity_map(R2, dec=1, title='Intensity Map',
                  valmin=None, valmax=None,):
    plt.figure(title, figsize=(8, 7))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(R2, dec), vmin=valmin, vmax=valmax,
                # cmap = 'Oranges', annot = True)
                cmap = 'Blues', annot = True)


def corr_circle(R, k1, k2, xLabel=None, yLabel=None,
              title='Correlation circle', valMin=-1, valMax=1):
    plt.figure(title, figsize=(10, 10))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    if xLabel == None or yLabel == None:
        if isinstance(R, pd.DataFrame):
            plt.xlabel(xlabel=R.columns[k1], fontsize=14,
                       color='k', verticalalignment='top')
            plt.ylabel(ylabel=R.columns[k2], fontsize=14,
                       color='k', verticalalignment='bottom')
        else:       # isinstance(R, np.ndarray):
            plt.xlabel(xlabel='Component ' + str(k1 + 1), fontsize=14,
                       color='k', verticalalignment='top')
            plt.ylabel(ylabel='Component ' + str(k2 + 1), fontsize=14,
                       color='k', verticalalignment='bottom')
    else:
        plt.xlabel(xlabel=xLabel, fontsize=14, color='k', verticalalignment='top')
        plt.ylabel(ylabel=yLabel, fontsize=14, color='k', verticalalignment='bottom')
    T = [t for t in np.arange(0, np.math.pi * 2, 0.01)]
    X = [np.cos(t) for t in T]
    Y = [np.sin(t) for t in T]
    plt.plot(X, Y)
    plt.axhline(0, color='g')
    plt.axvline(0, color='g')
    if isinstance(R, pd.DataFrame):
        plt.scatter(R.iloc[:, k1], R.iloc[:, k2], c='r', vmin=valMin, vmax=valMax)
        for i in range(len(R)):
            plt.text(R.iloc[i, k1], R.iloc[i, k2], R.index[i])
    else:   # isinstance(R, np.ndarray):
        plt.scatter(R[:, k1], R[:, k2], c='r', vmin=valMin, vmax=valMax)
        for i in range(len(R)):
            # position of the text on the graphic (x, y) and the actual text
            plt.text(R[i, k1], R[i, k2], '(' +
                     str(np.round(R[i, k1], 1)) +
                     ', ' +
                     str(np.round(R[i, k2], 1)) +
                     ')')


def corr_circle_2(t1, t2, xlabel='X', ylabel='Y',
                  title='Correlation circle',
                  s1='Set X', s2='Set Y'):
    f = plt.figure(figsize=(6, 6))
    ax = f.add_subplot(1, 1, 1)
    x = [v for v in np.arange(0, np.pi * 2, 0.01)]
    cosx = np.cos(x)
    sinx = np.sin(x)
    ax.plot(cosx, sinx)
    ax.axhline(0, color='k')
    ax.axvline(0, color='k')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(t1.iloc[:, 0], t1.iloc[:, 1], c='r', label=s1)
    ax.scatter(t2.iloc[:, 0], t2.iloc[:, 1], c='b', label=s2)
    ax.legend()
    n = len(t1)
    m = len(t2)
    for i in range(n):
        ax.text(t1.iloc[i, 0], t1.iloc[i, 1], t1.index[i])
    for i in range(m):
        ax.text(t2.iloc[i, 0], t2.iloc[i, 1], t2.index[i])


# building the correlation circle
# including concentric circles for the quartiles
def corr_circle_quartiles(R, k1, k2, radius=1, con=False, xLabel=None, yLabel=None,
                 title='Correlation circle with quartiles', valMin=-1, valMax=1):
    plt.figure(title, figsize=(10, 10))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    if xLabel == None or yLabel == None:
        if isinstance(R, pd.DataFrame):
            plt.xlabel(xlabel=R.columns[k1], fontsize=14,
                       color='k', verticalalignment='top')
            plt.ylabel(ylabel=R.columns[k2], fontsize=14,
                       color='k', verticalalignment='bottom')
        else:       # isinstance(R, np.ndarray):
            plt.xlabel(xlabel='Component ' + str(k1 + 1), fontsize=14,
                       color='k', verticalalignment='top')
            plt.ylabel(ylabel='Component ' + str(k2 + 1), fontsize=14,
                       color='k', verticalalignment='bottom')
    else:
        plt.xlabel(xlabel=xLabel, fontsize=14, color='k', verticalalignment='top')
        plt.ylabel(ylabel=yLabel, fontsize=14, color='k', verticalalignment='bottom')

    # building the circle of correlations, of radius 1,
    # with the center at the origin of the coordinate axes
    # generate a list of values for an angle that
    # takes values around the circle
    T = [t for t in np.arange(0, np.pi * 2, 0.01)]
    X = [np.cos(t) * radius for t in T]  # f(t) = cos(t)
    Y = [np.sin(t) * radius for t in T]  # f(t) = sin(t)
    plt.plot(X, Y)

    if con:
        for k in range(1, 3 + 1):
            X = [np.cos(t) * 0.25 * k * radius for t in T]
            Y = [np.sin(t) * 0.25 * k * radius for t in T]
            plt.plot(X, Y)

    plt.axhline(0, color='g')
    plt.axvline(0, color='g')
    if isinstance(R, pd.DataFrame):
        plt.scatter(R.iloc[:, k1], R.iloc[:, k2], c='r', vmin=valMin, vmax=valMax)
        for i in range(len(R)):
            plt.text(R.iloc[i, k1], R.iloc[i, k2], R.index[i])
    else:   # isinstance(R, np.ndarray):
        plt.scatter(R[:, k1], R[:, k2], c='r', vmin=valMin, vmax=valMax)
        for i in range(len(R)):
            plt.text(R[i, k1], R[i, k2], '(' +
                     str(np.round(R[i, k1], 1)) +
                     ', ' +
                     str(np.round(R[i, k2], 1)) +
                     ')')


def multi_histogram(X, width=0.8, title='Multi-histogram', xLabel='Observations',
              yLabel='Values of the variables'):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    plt.xlabel(xlabel=xLabel, fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(ylabel=yLabel, fontsize=14, color='k', verticalalignment='bottom')

    if isinstance(X, pd.DataFrame):
        noOfCols = X.values.shape[1]
        noOfLines = X.values.shape[0]
        width /= noOfCols

        ind = np.arange(noOfLines)
        for j in range(noOfCols):
            plt.bar(x=ind + (width * j), height=X.iloc[:, j].values,
                    width=width, label=X.columns[j], align='center')
        plt.xticks(ind + width * (noOfCols - 1) / 2.0, X.index[:])
    if isinstance(X, np.ndarray):
        noOfCols = X.shape[1]
        noOfLines = X.shape[0]
        width /= noOfCols

        ind = np.arange(noOfLines)
        for j in range(noOfCols):
            plt.bar(x=ind + (width * j), height=X[:, j],
                    width=width, label='Var' + str(j + 1), align='center')
        plt.xticks(ind + width * (noOfCols - 1) / 2.0,
                   ('Set' + str(i + 1) for i in range(noOfLines)))

    plt.legend(loc='best')
    return None


def multi_histogram_df(tabel, width=0.8, title='Multi-histograma', xLabel='Observations',
                yLabel='Values of the variables'):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    plt.xlabel(xlabel=xLabel, fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(ylabel=yLabel, fontsize=14, color='k', verticalalignment='bottom')

    noOfCols = tabel.values.shape[1]
    noOfLines = tabel.values.shape[0]
    width /= noOfCols

    ind = np.arange(noOfLines)
    for j in range(noOfCols):
        plt.bar(x=ind + (width * j), height=tabel.iloc[:, j].values,
                width=width, label=tabel.columns[j], align='center')
    plt.xticks(ind + width * (noOfCols - 1) / 2.0, tabel.index[:])
    plt.legend(loc='best')
    return None


def observations_cloud(tabel, label=None, xLabel='Observations', yLabel='Values of the variables',
                 title='Multi scatterplot'):
    f = plt.figure(title, figsize=(15, 11))
    f1 = f.add_subplot(1, 1, 1)
    f1.set_title(title, fontsize=14, color='k', verticalalignment='bottom')
    f1.set_xlabel(xlabel=xLabel, fontsize=14, color='k', verticalalignment='top')
    f1.set_ylabel(ylabel=yLabel, fontsize=14, color='k', verticalalignment='bottom')
    for j in range(len(tabel.columns)):
        f1.scatter(x=tabel.index[:], y=tabel.iloc[:, j].values,
                   label=tabel.columns[j], cmap='viridis')
    plt.legend(loc='best')


def variables_cloud(tabel, xLabel='Values of the variables', yLabel='Observations',
                 title='Multi scatterplot'):
    f = plt.figure(title, figsize=(15, 11))
    f1 = f.add_subplot(1, 1, 1)
    f1.set_title(title, fontsize=14, color='k', verticalalignment='bottom')
    f1.set_xlabel(xlabel=xLabel, fontsize=14, color='k', verticalalignment='top')
    f1.set_ylabel(ylabel=yLabel, fontsize=14, color='k', verticalalignment='bottom')
    for j in range(len(tabel.columns)):
        f1.scatter(x=tabel.iloc[:, j].values, y=tabel.index[:],
                   label=tabel.columns[j], cmap='viridis')
    plt.legend(loc='best')


def points_cloud(x, y, label=None, xLabel="", yLabel="", title='Scatterplot'):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(xlabel=xLabel, fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(ylabel=yLabel, fontsize=14, color='k', verticalalignment='bottom')
    plt.scatter(x=x, y=y, cmap='viridis')
    if label is not None:
        n = len(label)
        for i in range(n):
            plt.text(x[i], y[i], label[i])


# Plot discriminant scores and centers
def centre_points_cloud(x, y, g, labels_1, x1, y1, g1, labels_2,
                    title='Plot observations in the discriminant axes', lx='z1', ly='z2'):
    q = len(labels_2)
    # rainbowMap = plt.get_cmap('rainbow',q)
    f = plt.figure(title, figsize=(11, 8))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=14, color='k')
    ax.set_xlabel(xlabel=lx, fontsize=14, color='k')
    ax.set_ylabel(ylabel=ly, fontsize=14, color='k')
    sb.scatterplot(x=x, y=y, hue=g, ax=ax, hue_order=g1)
    sb.scatterplot(x=x1, y=y1, hue=g1, ax=ax, legend=False, marker='s',
                   s=200)
    for i in range(len(labels_1)):
        ax.text(x[i], y[i], labels_1[i])
    for i in range(len(labels_2)):
        ax.text(x1[i], y1[i], labels_2[i], fontsize=14)


def show():
    plt.show()