'''code for hfdata.ipynb
'''

import os
import glob
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from scipy import stats
from arch.unitroot import VarianceRatio
from datetime import datetime, timedelta, date

# disable warnings emitted by warnings.warn re aesthetics of post
warnings.filterwarnings('ignore')

# import HF datasets
def import_hf_data():
    dfFX = pd.read_csv("data/ALL.csv", engine='python', header=None)
    bigF = pd.read_csv("data/bigF.csv", index_col=False, header=0, engine='python')
    bigF = bigF.drop(columns=['Unnamed: 0'])
    dat2 = pd.read_csv("data/DEXJPUS.csv", engine='python', header=0) 
    return dfFX, bigF, dat2

# assemble time series
def hf_data(dfFX, bigF, dat2):
    # Set column headings for AAPL and JPM
    bigF.columns = ["Date","Ticker","TimeBarStart","OpenBarTime","OpenBidPrice",
                "OpenBidSize","OpenAskPrice","OpenAskSize","FirstTradeTime",
                "FirstTradePrice","FirstTradeSize","HighBidTime","HighBidPrice",
                "HighBidSize","HighAskTime","HighAskPrice","HighAskSize",
                "HighTradeTime","HighTradePrice","HighTradeSize","LowBidTime",
                "LowBidPrice","LowBidSize","LowAskTime","LowAskPrice",
                "LowAskSize","LowTradeTime","LowTradePrice","LowTradeSize",
                "CloseBarTime","CloseBidPrice","CloseBidSize","CloseAskPrice",
                "CloseAskSize","LastTradeTime","LastTradePrice","LastTradeSize",
                "MinSpread","MaxSpread","CancelSize","VolumeWeightPrice",
                "NBBOQuoteCount","TradeAtBid","TradeAtBidMid","TradeAtMid",
                "TradeAtMidAsk","TradeAtAsk","TradeAtCrossOrLocked","Volume",
                "TotalTrades","FinraVolume","FinraVolumeWeightPrice",
                "UptickVolume","DowntickVolume","RepeatUptickVolume",
                "RepeatDowntickVolume","UnknownTickVolume","TradeToMidVolWeight",
                "TradeToMidVolWeightRelative","TimeWeightBid","TimeWeightAsk"]
    
    # Set a date-time index, using OpenBarTime
    bigF['DateTimeIndex'] = pd.to_datetime(bigF['Date'].astype(str)) + pd.to_timedelta(bigF['OpenBarTime'].astype(str))
    bigF = bigF.set_index('DateTimeIndex')
    
    # Set datatypes and add separate Date and Time columns in case useful later
    bigF['Ticker'] = bigF.Ticker.astype(str)
    bigF['CloseBidSize'] = bigF.CloseBidSize.astype(float)
    bigF['CloseAskSize'] = bigF.CloseAskSize.astype(float)
    bigF['CloseBidPrice'] = bigF.CloseBidPrice.astype(float)
    bigF['CloseAskPrice'] = bigF.CloseAskPrice.astype(float)
    
    # Reduce bigF to smF
    smF = bigF[['Ticker','CloseBidSize','CloseAskSize','CloseBidPrice',
                'CloseAskPrice']].copy()   
    smF['Date'] = pd.to_datetime(bigF['Date'].astype(str))
    smF['Time'] = pd.to_timedelta(bigF['OpenBarTime'].astype(str))
    smF['DateTime'] = pd.to_datetime(bigF['Date'].astype(str)) + pd.to_timedelta(bigF['OpenBarTime'].astype(str))
    
    # Compute WeightedMidPrice using the closing prices per analysis
    smF['WeightedMidPrice'] = ((smF['CloseBidSize']*smF['CloseAskPrice']) + (smF['CloseAskSize']*smF['CloseBidPrice'])) / (smF['CloseBidSize'] + smF['CloseAskSize'])
    
    # Raw returns
    AAPL_rr = smF.loc[smF['Ticker'] == "AAPL"]
    AAPL_rr = AAPL_rr['WeightedMidPrice'] - AAPL_rr['WeightedMidPrice'].shift(1)
    AAPL_rr = AAPL_rr[AAPL_rr.notna()].copy()
    AAPL_rr = AAPL_rr[AAPL_rr != 0].copy()
    JPM_rr = smF.loc[smF['Ticker'] == "JPM"]
    JPM_rr = JPM_rr['WeightedMidPrice'] - JPM_rr['WeightedMidPrice'].shift(1)
    JPM_rr = JPM_rr[JPM_rr.notna()].copy()
    JPM_rr = JPM_rr[JPM_rr != 0].copy()
    
    # Log returns
    AAPL_lr = smF.loc[smF['Ticker'] == "AAPL"]
    AAPL_lr = np.log(AAPL_lr['WeightedMidPrice'].astype(float))
    AAPL_lr = AAPL_lr - AAPL_lr.shift(1)
    AAPL_lr = AAPL_lr[AAPL_lr.notna()].copy()
    AAPL_lr = AAPL_lr[AAPL_lr != 0].copy()
    JPM_lr = smF.loc[smF['Ticker'] == "JPM"]
    JPM_lr = np.log(JPM_lr['WeightedMidPrice'].astype(float))
    JPM_lr = JPM_lr - JPM_lr.shift(1)
    JPM_lr = JPM_lr[JPM_lr.notna()].copy()
    JPM_lr = JPM_lr[JPM_lr != 0].copy()
    
    # Remove outliers
    Q1l = AAPL_lr.quantile(0.001)   
    Q3l = AAPL_lr.quantile(0.999)   
    IQl = Q3l - Q1l
    Q1r = AAPL_rr.quantile(0.001)   
    Q3r = AAPL_rr.quantile(0.999)   
    IQr = Q3r - Q1r
    AAPL_lr = AAPL_lr[~((AAPL_lr < (Q1l - 1.5 * IQl)) | (AAPL_lr > (Q3l + 1.5 * IQl)))]
    AAPL_rr = AAPL_rr[~((AAPL_rr < (Q1r - 1.5 * IQr)) | (AAPL_rr > (Q3r + 1.5 * IQr)))]
    JPM_lr = JPM_lr[~((JPM_lr < (Q1l - 1.5 * IQl)) | (JPM_lr > (Q3l + 1.5 * IQl)))]
    JPM_rr = JPM_rr[~((JPM_rr < (Q1r - 1.5 * IQr)) | (JPM_rr > (Q3r + 1.5 * IQr)))]
    
    # log returns only for models split into estimate (E=60%) and out-of-forecast (F=40%)
    AAPL = AAPL_lr.to_numpy(copy=True)
    JPM = JPM_lr.to_numpy(copy=True)
    aaplE = AAPL[0:269618,]
    aaplF = AAPL[269618:,]
    jpmE = JPM[0:200363,]
    jpmF = JPM[200363:,]
    aaplE = aaplE[:,np.newaxis]
    aaplF = aaplF[:,np.newaxis]
    jpmE = jpmE[:,np.newaxis]
    jpmF = jpmF[:,np.newaxis]

    # set column headings for EURUSD
    dfFX.columns = ['date','time','barOpenBid', 'barHighBid', 'barLowBid', 'barCloseBid','volume']
    totalEntries1y = dfFX.shape[0]

    # Set a date-time index, using time
    dfFX['DateTimeIndex'] = pd.to_datetime(dfFX['date'].astype(str)) + pd.to_timedelta(dfFX['time'].astype(str))
    dfFX = dfFX.set_index('DateTimeIndex')

    # Set datatypes and add separate Date and Time columns in case useful later
    dfFX['barOpenBid'] = dfFX.barOpenBid.astype(float)
    dfFX['barHighBid'] = dfFX.barHighBid.astype(float)
    dfFX['barLowBid'] = dfFX.barLowBid.astype(float)
    dfFX['barCloseBid'] = dfFX.barCloseBid.astype(float)
    dfFX['volume'] = dfFX.volume.astype(float)

    # Raw returns
    rrFX = dfFX['barCloseBid'] - dfFX['barCloseBid'].shift(1)
    rrFX = rrFX[rrFX.notna()].copy()
    rrFX = rrFX[rrFX != 0].copy()

    # log returns
    lrFX = np.log(dfFX['barCloseBid'].astype(float))
    lrFX = lrFX - lrFX.shift(1)
    lrFX = lrFX[lrFX.notna()].copy()
    lrFX = lrFX[lrFX != 0].copy()

    # Remove outliers
    Q1l = lrFX.quantile(0.0001)  
    Q3l = lrFX.quantile(0.9999)  
    IQl = Q3l - Q1l
    Q1r = rrFX.quantile(0.0001)   
    Q3r = rrFX.quantile(0.9999)   
    IQr = Q3r - Q1r
    lrFX = lrFX[~((lrFX < (Q1l - 1.5 * IQl)) | (lrFX > (Q3l + 1.5 * IQl)))]
    rrFX = rrFX[~((rrFX < (Q1r - 1.5 * IQr)) | (rrFX > (Q3r + 1.5 * IQr)))]

    # log returns only for models split into estimate (E=60%) and out-of-forecast (F=40%)
    FX = lrFX.to_numpy(copy=True)
    fxE = FX[0:457419,]
    fxF = FX[457419:,]
    fxE = fxE[:,np.newaxis]
    fxF = fxF[:,np.newaxis]

    # set variables for benchmark simulated dataset based on MSM vol framework
    kbar = 4
    b = 6
    m0 = 1.6
    gamma_kbar = 0.8
    sig = 2/np.sqrt(252)    
    T = 7087
    E = np.rint(0.6*T).astype(int)            
    
    # simulated daily returns using MSM vol framework:
    dat1 = simulate_data(b,m0,gamma_kbar,sig,kbar,T)
    dat1E = dat1[0:E,]
    dat1F = dat1[E:,]

    # DEXJPUS benchmark daily returns dataset 
    dat2 = dat2.loc[dat2.DEXJPUS != "."].DEXJPUS.astype(float)
    dat2 = np.array(dat2)
    dat2_rtn = dat2[0:-1]
    dat2 = np.log(dat2[1:])-np.log(dat2[0:-1])
    dat2 = dat2[dat2 != 0]
    dat2 = dat2[:,np.newaxis]
    dat2E = dat2[0:E,]
    dat2F = dat2[E:,]

    return (bigF, smF, AAPL_rr, JPM_rr, AAPL_lr, JPM_lr, aaplE, aaplF, jpmE, 
            jpmF, dfFX, rrFX, lrFX, fxE, fxF, dat1, dat1E, dat1F, dat2, 
            dat2_rtn, dat2E, dat2F)

# simulate benchmark vol dataset using Markov-Switching Multifractal parameters
def simulate_data(b,m0,gamma_kbar,sig,kbar,T):
    m0 = m0
    m1 = 2-m0
    g_s = np.zeros(kbar)
    M_s = np.zeros((kbar,T))
    g_s[0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
    for i in range(1,kbar):
        g_s[i] = 1-(1-g_s[0])**(b**(i))
    for j in range(kbar):
        M_s[j,:] = np.random.binomial(1,g_s[j],T)
    dat = np.zeros(T)
    tmp = (M_s[:,0]==1)*m1+(M_s[:,0]==0)*m0
    dat[0] = np.prod(tmp)
    for k in range(1,T):
        for j in range(kbar):
            if M_s[j,k]==1:
                tmp[j] = np.random.choice([m0,m1],1,p = [0.5,0.5])
        dat[k] = np.prod(tmp)
    dat = np.sqrt(dat)*sig* np.random.normal(size = T)   # VOL TIME SCALING
    dat = dat.reshape(-1,1)
    
    return dat

# Autocorrelation computation using numpy.corrcoef
def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))

# Set up squared returns for autocorrelograms
def autocorr_gram(smF, lrFX):

    # Set up | NB: lags equate to 1-day
    n_lags_a = 798      # 1 day, NB | i.shape = 353545, j.shape = 349845, k.shape = 449398
    n_lags_j = 593      # 1 day, NB | m.shape = 264334, n.shape = 257702, p.shape = 334048
    n_lags_fx = 1354    # 1 day, NB | q.shape = 762381, r.shape = 762381, s.shape = 762381
    sig = 0.05          # sig = significance level, which I set at 0.05
    zero = False        # Flag re to include (or not) the 0-lag autocorrelation. 

    # AAPL
    i = smF.loc[smF['Ticker'] == "AAPL"]     
    i = np.log(i['CloseBidPrice'].astype(float))
    i = i - i.shift(1)
    i = i[i.notna()].copy()
    i = i[i != 0].copy()

    j = smF.loc[smF['Ticker'] == "AAPL"]
    j = np.log(j['CloseAskPrice'].astype(float))
    j = j - j.shift(1)
    j = j[j.notna()].copy()
    j = j[j != 0].copy()

    k = smF.loc[smF['Ticker'] == "AAPL"]
    k = np.log(k['WeightedMidPrice'].astype(float))
    k = k - k.shift(1)
    k = k[k.notna()].copy()
    k = k[k != 0].copy()

    # JPM
    m = smF.loc[smF['Ticker'] == "JPM"]
    m = np.log(m['CloseBidPrice'].astype(float))
    m = m - m.shift(1)
    m = m[m.notna()].copy()
    m = m[m != 0].copy()

    n = smF.loc[smF['Ticker'] == "JPM"]
    n = np.log(n['CloseAskPrice'].astype(float))
    n = n - n.shift(1)
    n = n[n.notna()].copy()
    n = n[n != 0].copy()

    p = smF.loc[smF['Ticker'] == "JPM"]
    p = np.log(p['WeightedMidPrice'].astype(float))
    p = p - p.shift(1)
    p = p[p.notna()].copy()
    p = p[p != 0].copy()

    # EURUSD
    q = lrFX

    r = lrFX
    r = r[250000:,]     # arbitrarily select a sub-population

    s = lrFX
    Q1l = s.quantile(0.001)   
    Q3l = s.quantile(0.999)   
    IQl = Q3l - Q1l
    s = s[~((s < (Q1l - 1.5 * IQl)) | (s > (Q3l + 1.5 * IQl)))]

    return i, j, k, m, n, p, q, r, s, n_lags_a, n_lags_j,n_lags_fx, sig

# autocorrelations re diurnal patterns and high resolution views
def autocorr_highres(smF):
    # AAPL
    u = smF.loc[smF['Ticker'] == "JPM"]     
    u = np.log(u['CloseBidPrice'].astype(float))
    u = u - u.shift(1)
    u = u[u.notna()].copy()
    u = u[u != 0].copy()
    u = u['2020-03-31 00:00:01':'2020-03-31 23:59:59']

    v = smF.loc[smF['Ticker'] == "JPM"]
    v = np.log(v['CloseAskPrice'].astype(float))
    v = v - v.shift(1)
    v = v[v.notna()].copy()
    v = v[v != 0].copy()
    v = v['2020-03-31 00:00:01':'2020-03-31 23:59:59']

    w = smF.loc[smF['Ticker'] == "JPM"]
    w = np.log(w['WeightedMidPrice'].astype(float))
    w = w - w.shift(1)
    w = w[w.notna()].copy()
    w = w[w != 0].copy()
    w = w['2020-03-31 00:00:01':'2020-03-31 23:59:59']

    sig = 0.05          # sig = significance level, which I set at 0.05
    zero = False        # Flag re to include (or not) the 0-lag autocorrelation. 
    n_lags = 500

    return u, v, w, sig, zero, n_lags

# variance ratios
def vr_list(n_lags_fx, AAPL_lr, JPM_lr, lrFX, bigF, dfFX):

    # compute simple (i.e., not WMP) returns
    AAPL_lr_simple_OB = bigF.loc[bigF['Ticker'] == "AAPL"]
    AAPL_lr_simple_OB = np.log(AAPL_lr_simple_OB['HighAskPrice'].astype(float))
    AAPL_lr_simple_OB = AAPL_lr_simple_OB - AAPL_lr_simple_OB.shift(1)
    AAPL_lr_simple_OB = AAPL_lr_simple_OB[AAPL_lr_simple_OB.notna()].copy()
    AAPL_lr_simple_OB = AAPL_lr_simple_OB[AAPL_lr_simple_OB != 0].copy()

    JPM_lr_simple_OB = bigF.loc[bigF['Ticker'] == "JPM"]
    JPM_lr_simple_OB = np.log(JPM_lr_simple_OB['HighAskPrice'].astype(float))
    JPM_lr_simple_OB = JPM_lr_simple_OB - JPM_lr_simple_OB.shift(1)
    JPM_lr_simple_OB = JPM_lr_simple_OB[JPM_lr_simple_OB.notna()].copy()
    JPM_lr_simple_OB = JPM_lr_simple_OB[JPM_lr_simple_OB != 0].copy()

    # simple dataset for FX based on barHighBid
    lrFX_simple = np.log(dfFX['barHighBid'].astype(float))
    lrFX_simple = lrFX_simple - lrFX_simple.shift(1)
    lrFX_simple = lrFX_simple[lrFX_simple.notna()].copy()
    lrFX_simple = lrFX_simple[lrFX_simple != 0].copy()
    
    # set up the variance ratio charts
    vr_a = []
    vr_j = []
    vr_fx = []
    vr_a_simple = []
    vr_j_simple = []
    vr_fx_simple = []

    i_start = 60       # start at 60 to remove higher frequencies

    # write vr into 3 separate lists for OB data subsets
    for i in range (i_start, n_lags_fx):
        vr_a.append(VarianceRatio(AAPL_lr, i).vr)     
        vr_j.append(VarianceRatio(JPM_lr, i).vr)
        vr_fx.append(VarianceRatio(lrFX, i).vr)
        # simple based on OB datasets
        vr_a_simple.append(VarianceRatio(AAPL_lr_simple_OB, i).vr)
        vr_j_simple.append(VarianceRatio(JPM_lr_simple_OB, i).vr)
        vr_fx_simple.append(VarianceRatio(lrFX_simple, i).vr)

    # put AAPL, JPM, and FX lists into three dataframes 
    vr_list_a = pd.DataFrame(np.column_stack([vr_a_simple, vr_a]), 
                               columns=['AAPL-WMP', 'AAPL-simple'])

    vr_list_j = pd.DataFrame(np.column_stack([vr_j_simple, vr_j]), 
                               columns=['JPM-WMP', 'JPM-simple'])

    vr_list_fx = pd.DataFrame(np.column_stack([vr_fx, vr_fx_simple]), 
                               columns=['EURUSD-WMP', 'EURUSD-simple'])

    return (vr_list_a, vr_list_j, vr_list_fx, vr_a, vr_a_simple, vr_j, 
            vr_j_simple, vr_fx, vr_fx_simple)



