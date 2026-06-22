'''
Main program for Analysis of Canonical Correlations (CCA)
Input data: two data tables relating to the same observations
The junction between the tables is made based on the index attribute
'''
import pandas as pd
import numpy as np
import utilsCCA as utils
import graphicsCCA as graphics
import cca.CCA as cca


try:
    file_1 = './dataIN/EnergyProduction.csv'
    file_2 = './dataIN/EnergyConsumption.csv'
    t1 = pd.read_csv(file_1, index_col=0)
    vars_1 = [str(v) for v in t1.columns]
    t2 = pd.read_csv(file_2, index_col=0)
    vars_2 = [str(v) for v in t2.columns]

    # merge the tables
    t = t1.merge(t2, left_index=True, right_index=True)

    # fetch the data sets as numpy.ndarray
    x = t.iloc[:, 1:5].values
    y = t.iloc[:, 6:10].values
    utils.replaceNAN(x)
    utils.replaceNAN(y)
    varX = t.columns[1:5]
    varY = t.columns[6:10]
    obs = t1.index

    # build and apply CCA model
    modelCCA = cca.CCA(x, y)
    r, r2, z, u = modelCCA.getCanonicalRoots()

    n, p = np.shape(x)
    q = np.shape(y)[1]
    m = min(p, q)
    pd.DataFrame(data=z, index=obs,
        columns=['z'+str(i) for i in range(1, m+1)]).to_csv('./dataOUT/z.csv')
    pd.DataFrame(data=u, index=obs,
        columns=['u' + str(i) for i in range(1, m+1)]).to_csv('./dataOUT/u.csv')
    chi2Calc, chi2Tab, dof = modelCCA.chi2BartlettWilks_test()
    print(chi2Calc, chi2Tab, dof)

    # save the canonical correlations and the significance test
    root_index = ['root' + str(i) for i in range(1, m + 1)]
    chi2_df = pd.DataFrame(data={'r': r, 'r2': r * r,
            'chi2Tab': chi2Tab, 'chi2Calc': chi2Calc, 'dof': dof},
                              index=root_index)
    chi2_df.to_csv('./dataOUT/chi2.csv')
    chi2_ = pd.DataFrame(chi2_df['chi2Tab'])
    graphics.intensity_map(chi2_, dec=4,
                         title='Bartlett-Wilks test of significance')

    # determining the number of significant canonical roots
    if any(np.array(chi2Tab) > 0.01):
        m_ = len(np.where(np.array(chi2Tab) > 0.01)[0])
        print('The no. of significant cannonical roots:', m_)
    if m_ == 0:
        print("There are no significant canonical roots!")
        print("The data sets are independent.")
        exit(-1)

    m_ = m
    # calculation of the correlation between the significant
    # canonical variables and the observed variables
    r_xz = np.corrcoef(x, z[:, :m_], rowvar=False)[:p, p:]
    r_yz = np.corrcoef(y, z[:, :m_], rowvar=False)[:q, q:]
    r_xu = np.corrcoef(x, u[:, :m_], rowvar=False)[:p, p:]
    r_yu = np.corrcoef(y, u[:, :m_], rowvar=False)[:q, q:]
    r_xz2 = r_xz * r_xz
    r_yu2 = r_yu * r_yu
    r_xz_df = pd.DataFrame(r_xz,
                columns=['z'+str(i) for i in range(1, m_ + 1)], index=varX)
    r_yu_df = pd.DataFrame(r_yu,
                columns=['u'+str(i) for i in range(1, m_ + 1)], index=varY)
    r_xz2_df = pd.DataFrame(r_xz2,
                columns=['z'+str(i) for i in range(1, m_ + 1)], index=varX)
    r_yu2_df = pd.DataFrame(r_yu2,
                columns=['u'+str(i) for i in range(1, m_ + 1)], index=varY)
    r_xz_df.to_csv('./dataOUT/r_xz.csv')
    r_yu_df.to_csv('./dataOUT/r_yu.csv')
    graphics.correlogram(r_xz_df, valmin=-1, valmax=1,
                         title='Correlogram x-z')
    graphics.correlogram(r_yu_df, valmin=-1, valmax=1,
                         title='Correlogram y-u')
    graphics.intensity_map(r_xz2_df, dec=2, title='Common variance x-z')
    graphics.intensity_map(r_yu2_df, dec=2, title='Common variance y-u')

    # selection of canonical roots for interpretation and analysis
    select_root = root_index[:m_]
    i1 = root_index.index(select_root[0])
    labelX = select_root[0]
    for i in range(1, len(select_root)):
        i2 = root_index.index(select_root[i])
        labelY = select_root[i]
        graphics.corr_circle_2(t1=r_xz_df.iloc[:, [i1, i2]],
                t2=r_yu_df.iloc[:, [i1, i2]],
                xlabel=labelX, ylabel=labelY,
                title='Causal variables in the space of canonical roots')

    # calculation of the common and redundant variance
    vx = np.sum(r_xz * r_xz, axis=0)
    vy = np.sum(r_yu * r_yu, axis=0)
    sx = vx * r2[:m_]
    sy = vy * r2[:m_]

    # tabulation variance/redundancy
    t_vr = pd.DataFrame(data={
        'vx': vx, 'vy': vy, 'vx(%)': vx * 100 / p, 'vy(%)': vy * 100 / q,
        'sx': sx, 'sy': sy, 'sx(%)': sx * 100 / p, 'sy(%)': sy * 100 / q
    }, index=root_index[:m_])
    t_vr.to_csv('./dataOUT/var_red.csv')
    if n < 500:
        labelX = 'z1/u1'
        i1 = root_index.index(select_root[0])
        for i in range(1, len(select_root)):
            i2 = root_index.index(select_root[i])
            labelY = 'z' + str(i2 + 1) + '/u' + str(i2 + 1)
            graphics.biplot(z[:, [i1, i2]], u[:, [i1, i2]], labelX, labelY,
                      'Biplot observations in in the space of  ' + labelX + ' - ' + labelY,
                      list(obs),list(obs))

    # compute the sum of absolute differences between canonical roots
    diff = abs(z[:, :m_] - u[:, :m_])
    sumdiff = np.array(np.sum(diff, axis=1))
    # sumdiff = np.array(np.mean(diff, axis=1))
    # sumdiff = np.array(np.mean(diff, axis=1)).reshape((n,1))
    print(sumdiff)
    diff_df = pd.DataFrame(data={'differences': sumdiff}, index=t.index)
    # diff_df = pd.DataFrame(data=sumdiff, index=t.index, columns=['differences'])
    print(diff_df)
    diff_df.to_csv(path_or_buf='./dataOUT/diff.csv', index_label='Country')

    graphics.show()

except Exception as ex:
    print("Error!")
