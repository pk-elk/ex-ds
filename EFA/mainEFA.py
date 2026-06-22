import numpy as np
import pandas as pd
import utils as utl
import efa.EFA as aef
import factor_analyzer as fa
import Graphics as g
from sklearn.preprocessing import StandardScaler


tabel = pd.read_csv('dataIN/MortalityEU.csv', index_col=0, na_values=':')
print(tabel)

obsNume = tabel.index.values
varNume = tabel.columns.values
matrice_numerica = tabel.values

# TODO