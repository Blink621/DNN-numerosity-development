# Select numerosity unit based on r score and correlation

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from os.path import join as pjoin


def gaussian(x, *param):
    return param[0] * np.exp(-(x - param[1])**2 / (2 * param[2]**2))


filelist = ['act_fc1_relu.npy', 'act_fc2_relu.npy', 'act_fc3_softmax.npy']


act_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/activation' 
out_path = '/nfs/s2/userhome/zhouming/workingdir/numerosity/out/numerosity_unit' 

numlist = np.asarray(range(1,33))
for filename in filelist:
    act = np.load(pjoin(act_path, filename))
    exloc = []
    for fmap in range(np.shape(act)[1]):
        for i in range(np.shape(act)[2]):
            for j in range(np.shape(act)[3]):
                exloc.append([fmap, i, j])
    
    idlist80 = []
    idlist85 = []
    idlist90 = []
    idlist95 = []
    nid = 0
    for locxyz in exloc:
        nid += 1
        col1 = act[:, locxyz[0], locxyz[1], locxyz[2]]
        col2 = np.repeat([1,2,3], np.shape(act)[0] / 3, axis=0)
        col3 = np.tile(np.repeat(numlist, 600),3)
        col4 = np.zeros(np.shape(act)[0])
        col5 = np.repeat(nid, np.shape(act)[0], axis=0)
        mat = np.zeros((np.shape(act)[0], 5))
        mat[:,0] = col1
        mat[:,1] = col2
        mat[:,2] = col3
        mat[:,3] = col4
        mat[:,4] = col5
        df = pd.DataFrame(mat)
        df.columns = ['act', 'dset', 'num', 'pn', 'id']
    
        dfmean = df.groupby(df['num']).mean()
        dfactm = dfmean['act']
        maxdfmean = np.max(dfmean['act'])
        if maxdfmean == 0:
            pn = 0
        else:
            pn = dfmean[dfmean.loc[:,'act'].isin([maxdfmean])].index[0]
        df['pn'] = np.repeat(pn, np.shape(act)[0])
# %%
        judge = np.min(df['act'].groupby(df['dset']).max())
        if judge != 0:
            dfm = df.groupby(df['num']).mean()['act']
            dfmn = (dfm - dfm.min()) / (dfm.max() - dfm.min())
            try:
                popt, pcov = curve_fit(gaussian, np.log2(numlist), dfmn, p0=[5, 5, 5], bounds=([0, 0, 0], [np.inf, np.log2(33), np.inf]), maxfev=5000000)
                r2 = r2_score(dfmn, gaussian(np.log2(numlist), *popt))
                actnum = []
                actnumn = []
                for index in range(3):
                    dfnum = df[df['dset'] == index+1]
                    dfnum = dfnum.groupby(dfnum['num']).mean()['act']
                    dfnumn = (dfnum - dfnum.min()) / (dfnum.max() - dfnum.min())
                    actnum.append(dfnum)
                    actnumn.append(dfnumn)
                sim = (np.corrcoef(actnum).sum() - 3) / 6
                if (r2 > 0.8) & (sim > 0.8):
                    idlist80.append([nid, df['pn'].max(), r2, popt[0], popt[1], popt[2], sim])
                if (r2 > 0.85) & (sim > 0.85):
                    idlist85.append([nid, df['pn'].max(), r2, popt[0], popt[1], popt[2], sim])
                if (r2 > 0.9) & (sim > 0.9):
                    idlist90.append([nid, df['pn'].max(), r2, popt[0], popt[1], popt[2], sim])
                if (r2 > 0.95) & (sim > 0.95):
                    idlist95.append([nid, df['pn'].max(), r2, popt[0], popt[1], popt[2], sim])
            except:
                print(df['pn'].max())
        print(f'Finish computing {nid} units of {filename}')
    
    df_80 = pd.DataFrame(idlist80, columns=['unit_id', 'pre_num', 'r2', 'gau_amplitude', 'gau_miu', 'gau_sigma', 'corr'])
    df_85 = pd.DataFrame(idlist80, columns=['unit_id', 'pre_num', 'r2', 'gau_amplitude', 'gau_miu', 'gau_sigma', 'corr'])
    df_90 = pd.DataFrame(idlist80, columns=['unit_id', 'pre_num', 'r2', 'gau_amplitude', 'gau_miu', 'gau_sigma', 'corr'])
    df_95 = pd.DataFrame(idlist80, columns=['unit_id', 'pre_num', 'r2', 'gau_amplitude', 'gau_miu', 'gau_sigma', 'corr'])
    
    df_80.to_csv(pjoin(out_path, 'id80_' + filename[4:-4] + '.csv'))
    df_85.to_csv(pjoin(out_path, 'id85_' + filename[4:-4] + '.csv'))
    df_90.to_csv(pjoin(out_path, 'id90_' + filename[4:-4] + '.csv'))
    df_95.to_csv(pjoin(out_path, 'id95_' + filename[4:-4] + '.csv'))
    
