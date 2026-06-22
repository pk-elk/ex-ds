import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd


# Pthon Graph Gallery
# https://python-graph-gallery.com/


# create the correlogram graphic
# assume that we receive a numpy.ndarray or
# pandas.DataFrame
def correlogram(R2=None, dec=2, title='Correlogram',
                valMin=-1, valMax=1):
    plt.figure(num=title, figsize=(12, 8))
    plt.title(label=title, fontsize=12,
              verticalalignment='bottom',
              color='Blue')
    sb.heatmap(data=np.round(a=R2, decimals=dec), vmin=valMin, vmax=valMax,
               cmap='bwr', annot=True)


def link_intensity(matrix=None, dec=2, title='Link Intensity',
                color='Oranges'):
    plt.figure(num=title, figsize=(12, 8))
    plt.title(label=title, fontsize=12,
              verticalalignment='bottom',
              color='Blue')
    sb.heatmap(data=np.round(a=matrix, decimals=dec),
               cmap=color, annot=True)


def correlation_circle(R2=None, V1=0, V2=1,
                    dec=2, title='Correlation circle'):
    plt.figure(num=title, figsize=(8, 7))
    plt.title(label=title+' between '+'C'+str(V1+1)+' and '+'C'+str(V2+1),
              fontsize=12,
              verticalalignment='bottom',
              color='Blue')
    # generate points (x, y) for drawing a circle
    theta = [t for t in np.arange(start=0, stop=2*np.pi, step=0.01)]
    x = [np.cos(t) for t in theta]
    y = [np.sin(t) for t in theta]
    plt.plot(x, y)
    plt.axhline(y=0, color='Green')
    plt.axvline(x=0, color='Green')

    if isinstance(R2, np.ndarray):
        plt.xlabel(xlabel='Variable ' + str(V1+1), fontsize=10,
                  verticalalignment='top',
                  color='Blue')
        plt.ylabel(ylabel='Variable ' + str(V2+1), fontsize=10,
                   verticalalignment='bottom',
                   color='Blue')
        plt.scatter(x=R2[:, V1], y=R2[:, V2], color='Red')
        for i in range(R2.shape[0]):
            # plt.text(x=R2[i, V1], y=R2[i, V2], s='text')
            plt.text(x=R2[i, V1], y=R2[i, V2],
                s='(' + str(np.round(R2[i, V1], decimals=dec)) + ', ' +
                str(np.round(R2[i, V2], decimals=dec)) + ')', color='Black')
    elif isinstance(R2, pd.DataFrame):
        plt.xlabel(xlabel=R2.columns[V1], fontsize=10,
                  verticalalignment='top',
                  color='Blue')
        plt.ylabel(ylabel=R2.columns[V2], fontsize=10,
                   verticalalignment='bottom',
                   color='Blue')
        # plt.scatter(x=R2.iloc[:].iloc[V1], y=R2.iloc[:].iloc[V2], color='Blue')
        # plt.scatter(x=R2.values[:, V1], y=R2.values[:, V2], color='Blue')
        plt.scatter(x=R2.iloc[:, V1], y=R2.iloc[:, V2], color='Blue')
        for i in range(R2.index.size):
            plt.text(x=R2.iloc[i].iloc[V1], y=R2.iloc[i].iloc[V2], color='Black',
                     s=R2.index[i])
    else:
        raise Exception('Invalid input data type!')


def eigenvalues(val, title='Explained variance by the principal components'):
    plt.figure(num=title, figsize=(10, 7))
    plt.title(label=title, fontsize=12,
              verticalalignment='bottom',
              color='Blue')
    plt.xlabel(xlabel='Pricipal components', fontsize=10,
               verticalalignment='top',
               color='Blue')
    plt.ylabel(ylabel='Eigenvalues - explained variance', fontsize=10,
               verticalalignment='bottom',
               color='Blue')
    components = ['C'+str(i+1) for i in range(val.shape[0])]
    plt.plot(components, val, 'bo-')
    # plt.plot(components, val, 'b*-')
    # plt.plot(components, val, 'b>-')
    plt.axhline(y=1, color='Red')


def show():
    plt.show()
