import pandas as pd
import Functions as f
import Graphics as g
import pca.PCA as pca


table = pd.read_csv('./dataIN/Teritorial.csv', index_col=0)
print(table)

# no. of observations
n = table.index.size
print(n)
# the labels for the observations
obs = table.index.values
print(obs, type(obs), obs.shape)

# np. useful variables (columns)
m = table.columns[1:].size
# the labels for the variables
vars = table.columns[1:].values
print(vars, type(vars), vars.shape)

X = table[vars].values
print(X, type(X), X.shape)

# compute the X standardized
X_std = f.standardize(X)
print(X_std, type(X_std), X_std.shape)

# create a PCA instance
pca_model = pca.PCA(X)

# extract the correlation matrix
corr = pca_model.getCorr()
# print(corr)
# save the correlation matrix in a CSV file
# TODO

# extract the eigenvalues - explained variance by the principal components
alpha = pca_model.getAlpha()
g.eigenvalues(val=alpha)
# g.show()

# extract the principal components
comp = pca_model.getComponents()
# save the principal components in a CSV file
comp_df = pd.DataFrame(data=comp,
        index=(observation for observation in obs),
        columns=('C'+str(j+1) for j in range(comp.shape[1])))
comp_df.to_csv('./dataOUT/PrincipalComponents.csv')
g.link_intensity(matrix=comp_df, title='Principal components')

# get the factor loadings
Rxc = pca_model.getFactorLoadings()
print(Rxc, type(Rxc), Rxc.shape)
# save the matrix of factor loadings in a CSV file
Rxc_df = pd.DataFrame(data=Rxc,
        index=(variable for variable in vars),
        columns=('C'+str(j+1) for j in range(Rxc.shape[1])))
Rxc_df.to_csv('./dataOUT/FactorLoadings.csv')

# correlogram of factor loadings
g.correlogram(R2=Rxc_df)
# create teh correlation circle between first 2 principal components
g.correlation_circle(R2=Rxc_df)
# g.show()

# extract the scores (standardized principal components)
scores = pca_model.getScores()
# save the matrix of scores in a CSV file
scores_df = pd.DataFrame(data=scores,
        index=(observation for observation in obs),
        columns=('C'+str(j+1) for j in range(comp.shape[1])))
scores_df.to_csv('./dataOUT/Scores.csv')
g.link_intensity(matrix=scores_df, title='Matrix of scores',
                 color='Blues')
# g.show()

# extract the communalities
commun = pca_model.getCommunalities()
# save the matrix of communalities in a CSV file
commun_df = pd.DataFrame(data=commun,
         index=(variable for variable in vars),
         columns=('C' + str(j + 1) for j in range(Rxc.shape[1])))
# TODO
g.correlogram(R2=commun_df, title='Communalities')
g.show()