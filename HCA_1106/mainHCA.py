import pandas as pd
import numpy as np
import graphicsHCA as g
import utilsHCA as utl
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as hdist
import sklearn.decomposition as dec
import matplotlib as mpl


fileName = './dataIN/Indicators_EN.csv'

# mpl.rcParams['figure.max_open_warning'] = 50
# print(mpl.rcParams['figure.max_open_warning'])

table = pd.read_csv(fileName, index_col=0)
print(table)

obs = table.index.values
print(obs, type(obs), obs.shape)
n = len(obs)
print('No. of observations:', n)

vars = table.columns[1:].values
print(vars, type(vars), vars.shape)
m = len(vars)
print('No. of variables:', m)

table_nda = table[vars].values
# replace NAN
X = utl.replace_na(table_nda)

# standardize X
X_std = utl.standardise(X)
# save X_std as CSV file
X_std_df = pd.DataFrame(data=X_std, index=obs, columns=vars)
X_std_df.to_csv(path_or_buf='./dataOUT/X_std.csv')

# create a list of clustering methods
methods = list(hclust._LINKAGE_METHODS)
print(methods, type(methods))

# create a list of metrics
metrics = hdist._METRICS_NAMES
print(metrics, type(metrics))

# hierarchical clustering
h_1 = hclust.linkage(y=X_std, method='average', metric='euclidean')
print(h_1)
# compute the threshold, the junction at which
# the partition of maximum stability occurs
# the maxim number of junctions
threshold, j, k = utl.threshold(h_1)
print(threshold, j, k)

# create dendrogram graphic
g.dendrogram(h=h_1, labels=obs,
             title="Hierarchical classification (method='average', metric='euclidean')",
               threshold=threshold, colors=None)
# g.show()

# hierarchical clustering
h_2 = hclust.linkage(y=X_std, method='single', metric='cityblock')
print(h_2)
# compute the threshold, the junction at which
# the partition of maximum stability occurs
# the maxim number of junctions
threshold, j, k = utl.threshold(h_2)
print(threshold, j, k)

# create dendrogram graphic
g.dendrogram(h=h_2, labels=obs,
             title="Hierarchical classification (method='single', metric='cityblock')",
               threshold=threshold, colors=None)
# g.show()

# clustering the variables
# hierarchical clustering
h_3 = hclust.linkage(y=X_std.T, method='single', metric='correlation')
print(h_3)
# compute the threshold, the junction at which
# the partition of maximum stability occurs
# the maxim number of junctions
threshold, j, k = utl.threshold(h_3)
print(threshold, j, k)

# create dendrogram graphic
g.dendrogram(h=h_3, labels=vars,
             title="Hierarchical classification (method='single', metric='correlation')",
               threshold=threshold, colors=None)
g.show()

# TODO
