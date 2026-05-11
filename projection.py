#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:17:30 2020

@author: dan
"""
# %% Setup
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tools as tls
import datastore as ds
import Trend as tr
import datetime as dt
import xarray as xr
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pymc as pm
import arviz as az

dst = ds.dst  # Data Store, an object that loads and stores data in a common format
plt.style.use('clean')
pd.options.display.float_format = '{:.4f}'.format  # change print format
pd.options.display.width = 70
np.set_printoptions(precision=5, linewidth=70)

yn, mn, dn, en, sn = ['Year', 'Month', 'Data', 'Error', 'Smooth']

# %% Data Tools

def get_step(n: int, annual: bool=True):
    """ Return annual (default) or monthly warming curve due to an energy
        change based on Caldeira and Myhrvold 2013, Using 3 exponent model 
        with median values:
        
        th0    th1    th2    ta0    ta1    ta2
        .226   .354   .409   .586   7.15   237.7
        
        The `ta` values give the time frame for a 3-box model (land, upper 
        and lower ocean).
        
        Parameters
        ----------
        
        n: int
            Number of years of step change values
            
        annual: bool
            (Default True) True if annual, False if monthly
    """
    # f(t) = 1 - (th0 e^-t/ta0 + th1 e^-t/ta1 + th2 e^t/ta2)
    th = np.array([.226, .354, .409 ])  # median values
    ta = np.array([.586, 7.15, 237.7])
    period = {True: 1, False: 1/12}[annual]
    t = np.arange(0, n*period, period)
    ones = np.ones((len(th), len(t)))
    a = (ones * t).T / ta
    y = (th * np.exp(-a)).sum(axis=1)
    y = 1. - y.clip(0., 1.)
    return pd.Series(y, t, name='Step')

def get_simple_step(n: int, r: float)->pd.Series:
    ''' Return a function that approaches 1 over time in response to a step
        change in energy.
        
        Parameters
        ----------
        
        n: int
            Number of values
            
        r: float 
            Period of growth. Larger is slower.
    '''
    t = np.arange(n)
    y = 1. - np.exp(-t/r)
    return pd.Series(y, t, name='Step')

def convolve_step(data: pd.Series, 
                     step: pd.Series=None, 
                     annual: bool=True):
    """ Convolve data with forcing impulse.
        Assumes impulse has +/- values as well.

        data: Series
        impulse: DataFrame
        monthly: boolean  True if monthly data present, default False
    """
    if step is None:
        step = get_step(len(data), annual=annual)
    n = len(data)
    kernel = np.zeros(2 * n - 1)
    change = data.copy()
    # have to assume the year before data starts is 0,
    # otherwise mean shifts
    change.iloc[1:] -= data.iloc[:-1].values
    kernel[-n:] = step.values
    c = np.convolve(change.values, kernel, 'valid')
    result = pd.Series(index=data.index,
                       data=c[0:n])
    return result

def autocorr(data: pd.Series, N: int):
    '''
    Return the autocorrelation of the data for the first N+1 lags 

    Parameters
    ----------
    data : pd.Series
        data for the autocorrelation
    N : int
        Number of lags after 0 to calculate.

    Returns
    -------
    pd.Series of autocorrelations with lags as index.
    '''
    ac = pd.Series(index=np.arange(N+1))
    ac.index.name = 'Lags'
    name = 'Data'
    if hasattr(data, 'name'):
        name = data.name
    ac.name = f'Autocorrelation of {name}'
    for i in ac.index:
        ac[i] = data.autocorr(i)
    return ac

def test_stationarity(obs):
    """
    Test for stationarity using both the Augmented Dickey-Fuller 
    and KPSS tests
    
    Parameters
    ----------
    
    obs : array-like
        Data to test
    """
    
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import kpss
    
    print("\nResults of Dickey-Fuller Test:")
    print('Hypothesis: roots on unit circle (stationary)')
    dftest = adfuller(obs, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    
    print("\nResults of KPSS Test:")
    print("Hypothesis: series is trend stationary")
    kpsstest = kpss(obs, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)
    
    if dfoutput['p-value'] < 0.05:
        print('\n Roots not on unit circle (possibly non-stationary)')
    if kpss_output['p-value'] < 0.05:
        print('\n Series is not trend stationary')


def chow(res, res1, res2, k, p=0.05):
    ''' Return the result of the Chow f-statistic test to determine
        the likelihood that the residuals of a model of two sets
        of data can be better explained by the model of the combined
        data.
        
        res : (array-like) Residuals of the combined data
        
        res1, res2 : (array-like) Residuals of the two portions of
                     data each modeled separately.
                    
        k : (int) Number of parameters in the model
    '''
    lb = acorr_ljungbox(res, 24, k)
    lag = lb.lb_pvalue.idxmax()
    if lb.lb_pvalue[lag] < p:
        print('\nWARNING: Residual is autocorrelated and the Chow test ' + \
              'is unsuited.\n')
    
    Dn = k
    Dd = len(res1) + len(res2) - 2*k
    S = (res * res).sum()
    S12 = (res1 * res1).sum() + (res2 * res2).sum()
    C = ((S - S12)/Dn) / (S12 / Dd)
    p = stats.f.sf(C, Dn, Dd)
    r = pd.Series([C, p, Dn, Dd], 
                  index=['Chow', 'p', 'DoF1', 'DoF2'])
    return r

def fit(data, vars):
    ''' Fit a set of variables to data
    
        data and vars must have the same number of rows. Returns the scaled
        vars data, with a column containing a constant.
    '''
    n = len(data)
    A = np.hstack([vars.to_numpy(), np.ones((n, 1))])
    c = np.linalg.lstsq(A, data.to_numpy(), rcond=None)
    return c[0]

def date_string(d):
    """ Return date string from a tuple of (year, month)
    """
    return f'{d[0]}-{d[1]}-01'

def date_index(start, end, freq='MS'):
    """
    Get Pandas DateTime index with the supplied start and end dates

    Parameters
    ----------
    start : tuple(int, int)
        (year, month: 1-index).
    end : tuple(int, int)
        (year, month: 1-index).
    freq : string
        ['MS'|'Y'] : monthly start or yearly (default monthly)

    Returns
    -------
    DateTime index.
    """
    a = date_string(start)
    b = date_string(end)
    return pd.date_range(start=a, end=b, freq=freq)

def calc_volcano(end=None, annual=False):
    """
    Retrieve aerosol data and turn it into a volcanic aerosol forcing time
    series. Use seasonal solar variability, and albedo at each latitude to
    calculate.

    Parameters
    ----------
    end : tuple(year:int, month:int), or int, optional
        End date or year if it is earlier than data. The default is None.
    annual : bool, default False
        If True, return annual means.

    Returns
    -------
    cvol : pd.Series
        volcanic index (max = 1).

    """
    # Get aerosol data from the GloSSAC satellite instrument data set
    # This is in NetCDF format and requires a free account to access
    # Source: https://asdc.larc.nasa.gov/project/GloSSAC
    # A new version is available at the end of each year.
    url = 'Data/GloSSAC_V2.22.nc'
    gl = xr.open_dataset(url)
    aod = gl.Glossac_Aerosol_Optical_Depth[:,:,2]  # 525 nm
    # time index is integers in the form yyyymm
    time_index = aod.indexes['time']
    start = ((time_index[0] // 100), (time_index[0] % 100))
    last = ((time_index[-1] // 100), (time_index[-1] % 100))
    if not end:
        end = last
    elif hasattr(end, 'month'):
        end = (end.year, end.month)
    elif annual:
        end = (end, 12)
    dates = date_index(start, end)
    
    # new dataframe with date index
    vol_df = pd.DataFrame(index=dates, columns=aod.lat.values, 
                          dtype=np.float64)
    vol_df.iloc[:aod.values.shape[0],:] = aod.values
    # remove baseline aerosols from 1997 -2005 quiet period
    vol_df -= vol_df[(vol_df.index.year>=1997) & 
                     (vol_df.index.year<=2005)].mean()
    
    deg2rad = np.pi/180.0
    max_tilt_rad = 23.44 * deg2rad  # axial tilt
    # note month is 1-indexed
    tilt_rad = -max_tilt_rad * np.cos((dates.month - 2 + 21/31)/12 * 2 * np.pi)
    lat_rad = vol_df.columns.values * deg2rad
    true_lat_rad = np.add.outer(tilt_rad, lat_rad)
    adjust = np.maximum(0, np.cos(true_lat_rad))  # truncate negative values at 0
    adjust *= np.cos(lat_rad)  # area at polar latitudes less than at equator
    vol_df *= adjust

    # load planet albedo at 0.5° resolution for Dec and Jun, 2022
    # Satellite doesn't have albedo for regions in darkness.
    # Although Sep or Mar would work, more melting has occured for one hemisphere
    # Source: https://neo.gsfc.nasa.gov/view.php?datasetId=MCD43C3_M_BSA&year=2022
    url = 'Data/MCD43C3_M_BSA_2022-06-01_rgb_720x360.SS.CSV'
    albedo = pd.read_csv(url, sep=',', index_col=0, header=0, dtype=np.float64)
    url = 'Data/MCD43C3_M_BSA_2022-12-01_rgb_720x360.SS.CSV'
    south = pd.read_csv(url, sep=',', index_col=0, header=0, dtype=np.float64)
    albedo.loc[albedo.index < 0] = south.loc[south.index < 0]
    
    # Now calculate heating rate for land and ocean
    celsius2kelvin = 274.15
    area_ocean = .71  # ocean area of Earth
    area_land = 1. - area_ocean
    heat_ocean = .95  # amount of incoming heat energy stored in ocean
    heat_land = 1. - heat_ocean
    heat_per_area_ocean = heat_ocean / area_ocean
    heat_per_area_land = heat_land / area_land
    temperature_land = 8.6 + celsius2kelvin  # Berkely Earth
    temperature_global = 14.7 + celsius2kelvin  # Berkely Earth
    temperature_ocean = (temperature_global - temperature_land * area_land) \
                        / area_ocean
    heat_capacity_ocean = heat_per_area_ocean / temperature_ocean
    heat_capacity_land = heat_per_area_land / temperature_land
    ocean_vs_land_capacity = heat_capacity_ocean / heat_capacity_land  # 7.53
    # adjust albedo for ocean
    react = 1 - albedo
    # albedo of 9999 is for ocean or no data
    react[react < 0] = 1. / ocean_vs_land_capacity  # higher capacity less reactive
    react = react.mean(axis=1)
    r_df = pd.DataFrame(react, columns=['React'])
    # combine latitudes to match aerosol data
    r_df['lat'] = np.floor((r_df.index-5)/5)*5 + 7.5
    react = r_df.groupby('lat').mean()
    react = react / react.max()
    vol_df[:] *= react[-77.5:77.5].values.T
    vol = vol_df.mean(axis=1)
    vol.fillna(0, inplace=True)
    
    if annual:
        vol = vol.groupby(vol.index.year).mean()
    return vol

def fft(ds, mult=None):
    ''' 
    Return a DataFrame with the fft of the supplied residual series

    Parameters
    ----------
    ds : pd.Series
        Series to be analized. Any trends should be removed.
        
    mult : int
        Desired multiple of length to do the FFT. Data is zero-padded.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing columns for 'period', 'complex', 'value', 
        'angle'.

    '''
    if mult is None:
        n = len(ds)
    else:
        n = mult * len(ds)
    s = ds.to_numpy()
    f = np.fft.rfft(s, n)
    df = pd.DataFrame(index=np.arange(len(f)), dtype=np.float64)
    df['period'] = (n-1)/df.index
    df['complex'] = f
    df['value'] = np.abs(df.complex)
    df['angle'] = np.angle(df.complex)
    return df.iloc[1:]  # remove constant value

def fftn(ds, mult=None, verbose=False):
    '''
    Normalized FFT with noise baseline removed to identify signals. Calls
    `fft` to perform calculations.

    Parameters
    ----------
    ds : pandas.Series
        Signal data.
    mult : int, optional
        Multiple of data length. The default is None.
    verbose : bool, optional
        If True, plot chart of fft data. The default is False.

    Returns
    -------
    fft : pandas.Dataframe 
        Has columns 'period', 'complex', 'value', 
        'angle', 'base', and 'residual'.
    '''
    ft = fft(ds, mult)
    ft.value /= ft.value.max()
    nf = mult * len(ds)
    ft['base'] = tr.lowess(ft.value, f=.25)
    ft['residual'] = ft.value - ft.base
    if verbose:
        fm = ft.nlargest(3, 'residual')
        ax = new_axes('FFT', 
                      'Amplitude of Frequencies in Residual', 
                      'Amplitude relative to maximum')
        ax.set_xlabel(f'Cycles per {nf} Months')
        ax.plot(ft.value)
        ax.plot(ft.base)
        ax.plot(fm.value, 'k+', lw=7)
        plt.show()
    return ft

def var2lag(vars):
    '''
    Turn a series of model variables into an array of lags
    '''
    p = vars.copy()
    if 'sigma2' in p.index:
        p.drop('sigma2', inplace=True)
    if 'ma.L1' in p.index:
        p.drop('ma.L1', inplace=True)
    ix = list(p.index.str.split('L'))
    lags = np.array(ix)[:,1].astype(int)
    return lags

# %% Data Processing

def compile_vars(source='hadcrut', start='1990-01-01'):
    """ Return DataFrame containing temperature, trend line, detrended 
        temperature, and environmental factors that might affect temperature.
    """
    temp = ds.load_modern(source, annual=False)    
    end = temp.index[-1]
    start = pd.to_datetime(start) # Start year for analysis
    df = pd.DataFrame(index=temp.loc[start:end].index)
    df.index.name = source
    df.spec = ''  # avoid warning for setting columns
    df.spec = temp.spec
    tr.convertYear(df)  # add a fractional year column for calculations
    df['temp'] = temp.loc[start:end, dn]
    df['lowess'] = tr.lowess(df.temp, pts=15*12)  # this is the local trend, not a line
    df['flat'] = df.temp - df.lowess
 
    df['vol'] = calc_volcano(end, annual=False)
    df['solar'] = dst.solar(annual=False).loc[start:end]
    df.solar -= df.solar.mean()  # remove offset
    enso = dst.enso(annual=False).loc[start:end]
    enso -= enso.mean()
    df[enso.columns] = enso
        
    df.fillna(0, inplace=True)
    return df

def adjust_vars(df, adj, val):
    ''' Adjust the supplied columns using the requested adjustment
        method. Return the adjusted columns
        
        df: (DataFrame) Contains ONLY columns to be adjusted
        
        adj: (str) ['cmip'|'exp'|'lag'] Adjustment type for each 
             column
        
        val: (float or int) Value to use for each column
    '''
    n = len(df)
    cols = df.columns
    vars = pd.DataFrame(index=df.index, columns=cols)
    vars[cols] = 0.0
    for i, col in enumerate(cols):
        if adj == 'cmip':
            vars[col] = convolve_step(df[col], annual=False)
        elif adj == 'exp':
            if val == 'none':
                vars[col] = df[col].to_numpy()  # simple copy  
            elif val == 'cmip':
                vars[col] = convolve_step(df[col], annual=False)
            elif val < 0.:  # no change, equivalent to doing nothing
                vars[col] = df[col].to_numpy()  # simple copy
            elif val == 0.:  # use the cmip model
                vars[col] = convolve_step(df[col], annual=False)
            else:
                step = get_simple_step(n, val)
                vars[col] = convolve_step(df[col], step)
        elif adj == 'lag':
            val = int(val)
            vars.iloc[val:, i] = df[col].iloc[:n-val]
        else:  # no change
            vars[col] = df[col]
    vars.fillna(0, inplace=True)
    return vars  

def adjust_combined(df, adjs):
    cols = dict(vol=['vol'], solar=['solar'], 
                enso='N12 N3 N4 N34'.split())
    
    for col in cols.keys():
        c = cols[col]
        df[c] = adjust_vars(df[c], 'exp', adjs.exp[col])
        df[c] = adjust_vars(df[c], 'lag', adjs.lag[col])
    return df
        
def optimize_adjustments(df=None, passes=20):
    ''' Return a table with the optimal value for each variable
        with every adjustment type. Optimal is minimization of
        the standard deviation. 
        
        df: (DataFrame) Data along with the flattened residual, and 
            the natural influences to be optimized.
            
        passes: (int) Maximum number of times to find optimal values.
                The algorithm should stop when two passes have the same
                result, but oscillations could potentially arise.
    '''
    if df is None:
        df = compile_vars()
    var_cols = 'vol solar N12 N3 N4 N34'.split()
    cols = dict(vol=['vol'], solar=['solar'], 
                enso='N12 N3 N4 N34'.split())
    vals = dict(exp=np.arange(-.1, 10.15, 0.1),
                lag=np.arange(0, 13, dtype=int))
    outcome = pd.DataFrame(index=cols.keys(), 
                           columns='exp lag result'.split())
    outcome.result = 1.0
    outcome.lag = 0
    outcome.exp = -0.1
    previous = outcome.copy()
    tests = pd.Series(index=vals['lag'], data=9.)
    original = df.flat.std()
    for p in range(passes):
        for col in cols.keys():
            ''' 
            For each natural influence `col`, smooth using the
            exponential value `exp`. Then check each lag `lag` to find
            the value with the smallest standard deviation, keeping
            other influences the same change as in `previous`. This
            is checked against the value in `outcome` and replaced
            if lower. At the end, check if `outcome` was changed from
            `previous`. If changed, copy `outcome` to `previous` and do
            another pass.
            '''
            print(col)
            vars = df[var_cols].copy()
            remain = list(cols.keys())
            remain.remove(col)
            for r in remain:  # revert remaining cols to previous best
                vars[cols[r]] = adjust_vars(df[cols[r]], 'exp',
                                            previous.exp[r])
                vars[cols[r]] = adjust_vars(vars[cols[r]], 'lag',
                                            previous.lag[r])
            for exp in vals['exp']:
                rexp = adjust_vars(df[cols[col]], 'exp', exp)
                for lag in vals['lag']:
                    vars[cols[col]] = adjust_vars(rexp, 'lag', lag)
                    c = fit(df.flat, vars[var_cols])
                    vars[var_cols] *= c[:-1]
                    residual = df.flat - vars[var_cols].sum(axis=1) - c[-1]
                    tests[lag] = residual.std()
                tests /= original
                # find best lag
                m =tests.min(skipna=True)
                mi = tests.idxmin(skipna=True)
                if m < outcome.result[col]:
                    outcome.loc[col, 'result'] = m
                    outcome.loc[col, 'lag'] = mi  # best lag
                    outcome.loc[col, 'exp'] = exp # current exp value
                tests.loc[:] = 9.
        print(f'\n==== Pass {p} ====\n{outcome}\n')
        test = outcome.result.sum() - previous.result.sum()
        if abs(test) < .000001:
            break
        previous = outcome.copy()
    print(f'\n{outcome}\n')
    return outcome
    
def fit_vars(df=None, adjs=None, annual=False, verbose=False):
    # Important columns:
        # temp: temperature
        # lowess: local weighted trend calculated from the temperature
        # flat: temperature - lowess
        # vars: natural variation fit to the flat values
        # reduced: flat - vars
        # real: reduced + lowess
        # trend: linear trend of real
        # detrend: real - trend
    cols = dict(vol=['vol'], solar=['solar'], 
                enso='N12 N3 N4 N34'.split())
    influences = 'vol solar N12 N3 N4 N34'.split()
    if df is None:
        df = compile_vars()
    if adjs is None:
        adjs = optimize_adjustments(df)
    n = len(df)
    # adjust natural influences for best fit
    df = adjust_combined(df, adjs)
    # get trend line
    rate = 10
    m, b = np.polyfit(df.Year, df.temp, 1)  # conversion to numpy not needed
    print(f'Original trend was {m*rate:.3}°C/decade')
    sigma = df.flat.std()
    print(f'Original standard deviation was: {sigma:.4f}°C')
    c = fit(df.flat, df[influences])  # Simultaneous Least Squares Fit
    df[influences] *= c[:-1]
    df['enso'] = df[cols['enso']].sum(axis=1)
    df['vars'] = df[influences].sum(axis=1) + c[-1]  # all variables with offset
    df['reduced'] = df.flat - df.vars
    
    # Remove annual and biennual signals
    names = []
    idx = np.arange(len(df))
    tau = 2. * np.pi
    for per in [6, 12, 24]:
        for sig, f in zip(['Cos', 'Sin'], [np.cos, np.sin]):
            name = f'{sig}{per}'
            names.append(name)
            df[name] = f(tau * idx/per)
    c = fit(df.reduced, df[names])
    df[names] *= c[:-1]
    df['seasonal'] = df[names].sum(axis=1) + c[-1]
    df.vars += df.seasonal
    df.reduced -= df.seasonal

    # Remove other periodic signals from residual
    ft = fftn(df.reduced, mult=4, verbose=verbose)
    nf = 2 * len(ft)  # only half the values are returned. They're symmetric.
    ns = 3  # number of signals to look at
    fm = ft.nlargest(ns, 'residual')
    print(f'\nTop {len(fm)} residual frequencies:')
    print(fm)
    
    # Check if signal is strong enough to use
    threshold = 6.  # threshold above which sine signals will be used
    signal = fm.value.sum()
    noise = ft.value.sum() - signal
    signal /= ns
    noise /= (nf/2 - ns)
    snr = 10 * np.log10(signal / noise)  # S/N in dB
    print(f'Signal to Noise Ratio: {snr:.1f} dB')
    print(f'SNR Threshold is {threshold:.0f} dB\n')
    
    if snr > threshold:  # remove periodic signals
        t = np.arange(n)
        fr = 2. * np.pi / nf
        for s in fm.index:
            sname = f'Sine{s}'
            df[sname] = np.cos(fr * s * t + fm.angle[s])
        sines = df.columns[-ns:]
        c = fit(df.reduced, df[sines])
        df[sines] *= c[:-1]
        df['sine'] = df[sines].sum(axis=1) + c[-1]
        print('Sine coef:', c)
        print(f'Sine impact is ±{0.707 * abs(c).sum()/df.vars.std():.3f}σ\n')
        df.reduced -= df.sine
        df.vars += df.sine
    
    # calculate true trend line of reduced data
    df['real'] = df.reduced + df.lowess
    m, b = np.polyfit(df.Year, df.real, 1)
    df.fit = ''
    df.fit = {'slope':m, 'intercept':b}
    df['trend'] = m * df.Year + b
    df['detrend'] = df.real - df.trend

    nsigma = df.detrend.std()
    print(f'New standard deviation is: {nsigma:.4f}°C')
    print(f'Reduction of {(sigma-nsigma)/sigma*100:.1f}%')
    print(f'New slope is {(m*rate):.3f}°C/decade')
    r2 = tls.R2(df.flat.values, df.vars.values)
    print(f'R² value is {r2:.3f}')
    
    if annual:
        df = df.groupby(df.index.year).mean()
    return df  
    
def get_future(adjs):
    ''' Return a dataframe with 2016 to 2020 residual added to the end.
    '''
    df = fit_vars(adjs=adjs)
    model = df.loc['2016-01-01':'2019-12-01']
    start = df.index[0]
    end = '2028-01-01'
    xp = pd.date_range(start, end, freq='MS', inclusive='left')  # projection
    future = pd.DataFrame(index=xp)
    idf = df.index
    ifut = future.index[-48:]
    future.loc[idf, 'detrend'] = df.reduced.to_numpy()
    future.loc[ifut, 'detrend'] = model.reduced.to_numpy()
    m = (df.trend.iloc[-1] - df.trend.iloc[0]) / len(df)
    ixp = np.arange(len(future))
    future['trend'] = m * ixp + df.trend.iloc[0]
    future['real'] = future.trend + future.detrend
    future.spec = ''
    future.spec = df.spec
    return future
    
def get_AR(data, N, full=False):
    '''
    Determine model parameters for an AR(N) model given residual data. Based on 
    https://www.pymc.io/projects/examples/en/latest/time_series/AR.html
    
    Parameters
    ----------
    data : pandas.Series
        Data to model. Should be stationary.
    N : int | list
        Number of lag parameters to return or a list of specific lag parameters
        to return.
    full : bool (default False)
        If True returns entire model output as an xarray. Otherwise
        returns a dataframe with means and stds.

    Returns
    -------
    pandas.DataFrame of parameters.
    xarray if full==True
    '''
    def get_rho(xf):
        dims = ['draw','chain']
        sigma = xf.posterior.sigma.mean().to_numpy()
        p = xf.posterior.drop_vars('sigma')
        if len(p.data_vars)==1:  # single variable                
            df = p.rho.mean(dims).to_dataframe('mu')
            df['sigma'] = p.rho.std(dims).to_series()
        else:
            mu = p.mean(dims).to_pandas()
            ix = list(mu.index.str.split('_'))
            ixa = np.array(ix)[:,1].astype(int) + 1
            df = pd.DataFrame(index=ixa, data=mu.values, columns=['mu'])
            df['sigma'] = p.std(dims).to_pandas().values
            lstr = np.array(['ar.L'] * len(df))
            df['sm'] = lstr + df.index.astype(str)
        df.index.name = 'lags'
        df.in_sigma = float(sigma)
        df._metadata.extend('in_sigma')  # enforce copying
        
        return df
    
    # Using the pyMC bayesian modelling language to estimate values.
    # See https://www.pymc.io/projects/examples/en/latest/time_series/AR.html
    # for a complete explanation.
    with pm.Model() as model:
        rho = pm.Uniform('rho', -.5, .5, shape=N)
        sigma = pm.HalfNormal('sigma', .1)
        likelihood = pm.AR('y', observed=data,
                           rho=rho, sigma=sigma, constant=False,
                           init_dist=pm.Normal.dist(0, 10))
        idata = pm.sample(1000, tune=2000)
    
    df = get_rho(idata)
    thresh = df.sigma.std()
    az.plot_trace(idata, combined=True, 
                  lines=[('rho', {},(-thresh, thresh))])
    if full:
        return idata
        
    # Try the model again but limit the parameters found
    lags = df.loc[np.abs(df.mu)>=df.sigma].index
    while len(df) != len(lags):  # some lags were dropped
        rho = [0] * N
        with pm.Model() as model:
            for i in lags:
                rho[i] = pm.Uniform(f'rho_{i}', -.5, .5)
            sigma = pm.HalfNormal('sigma', .1)
            likelihood = pm.AR('y', observed=data,
                               rho=rho, sigma=sigma, constant=False,
                               init_dist=pm.Normal.dist(0, 10))
            idata = pm.sample(1000, tune=2000)
        df = get_rho(idata)
        lags = df.loc[np.abs(df.mu)>=df.sigma].index
        print(f'\n{df}\n')
        
    return df

def get_ARIMA(s, N, use_diff=True):
    '''
    Return the parameters of the ARIMA model given data and number of lags
    to look at. These lags are calculated by creating a model with all possible
    lags, then dropping lags with low significance until none can be dropped.

    Parameters
    ----------
    s : pd.Series
        Data to model.
    N : int
        Number of lags to look at.
    use_diff : bool, optional
        If True, use a difference model to ensure stationarity of the data. 
        The default is True.

    Returns
    -------
    res : TYPE
        Details of the ARIMA model.

    '''
    
    # Change index to a date range with a monthly start frequency. This
    # isn't strictly necessary, but it removes an annoying warning.
    obs = pd.Series(index=pd.date_range(s.index[0], s.index[-1], freq='MS'))
    obs[s.index] = s  # missing data will be NaN
    d, m = (1, 1) if use_diff else (0, 0)
    fixed = {'ma.L1': -1}
    last = 0
    lags = np.arange(N) + 1
    while last != len(lags):
        last = len(lags)
        model = sm.tsa.arima.ARIMA(obs, order=(lags, d, m), dates=obs.index,
                                   freq='MS', enforce_invertibility=False)
        fixed['ma.L1'] = -1.
        dropping = ['ma.L1', 'sigma2']
        if not use_diff:
            fixed = {}
            dropping = ['sigma2']
        with model.fix_params(fixed):
            res = model.fit()
        par = res.params
        sigma = par.sigma2**.5
        par.drop(index=dropping, inplace=True)
        # Only keep parameters that are statistically different than 0
        keep = par[np.abs(par)>(sigma*.85)]
        lags = var2lag(keep)
        print(f'\n{keep}')
        
    return res

def AR_process(input, params):
    '''
    Run the input through an autoregression process. 
    
    Paramters
    ---------
    input : array-like
        values that will have the AR process applied to them.
    
    params : pd.Series
        result parameters from a statsmodel model fit.
    '''
    if type(input) == pd.Series:
        series = True
        x = input.to_numpy()
    else:
        x = input
        series = False
    p = params.copy()
    if 'sigma2' in p.index:
        p.drop('sigma2', inplace=True)
    lags = var2lag(p)
    p = p.to_numpy()
    k = lags.max()
    y = np.zeros(len(x)+k)
    for i in range(k, len(y)):
        y[i] = y[i - lags] @ p + x[i-k]  # matrix multiply
    if series:
        return pd.Series(data=y[k:], index=input.index)
    else:
        return y[k:]
    
def decorrelate(input, params):
    '''
    Run the input through an inverse autoregression process. 
    
    Paramters
    ---------
    input : array-like
        values that will have the AR process applied to them.
    
    params : pd.Series
        result parameters from a statsmodel model fit.
    '''
    if type(input) == pd.Series:
        series = True
        x = input.to_numpy()
    else:
        x = input
        series = False
    p = params.copy()
    if 'sigma2' in p.index:
        p.drop('sigma2', inplace=True)
    lags = var2lag(p)
    k = max(lags)
    dparams = np.zeros(k+1)
    dparams[0] = 1.
    dparams[lags] = -p
    r = np.convolve(x, dparams)
    if series:
        return pd.Series(r[:len(x)], input.index)
    else:
        return r[:len(x)]

# %% Plotting helpers

def new_axes(name, title, ylabel):
    fig, ax = plt.subplots(1, 1, num=name, clear=True)
    tls.byline(ax)
    tls.titles(ax, title, ylabel)
    ax.axhline(color='k', alpha=.15)
    return ax

def new_fig_rows(name, title, ylabel, num=2, labels=None, alternate=True):
    fig, axs = plt.subplots(num, 1, num=name, clear=True, sharex=True)
    fig.subplots_adjust(hspace=0)
    tls.byline(axs[-1])
    tls.titles(axs[0], title, ylabel)
    if alternate:
        for i in range(1, len(axs), 2):
            axs[i].tick_params('y', labelright=True, labelleft=False,
                               right=True, left=False)
    if labels:
        for i in range(len(labels)):
            axs[i].text(0.01, 0.97, labels[i], weight='bold', va='top',
                        transform=axs[i].transAxes)
    for ax in axs:
        ax.axhline(color='k', alpha=.15)
            
    return axs

def plot_one(ax, data, sigma, years=None, labels=None):
    """ Plot one axes given a Pandas Series as data
    """
    annual = not hasattr(data.index, 'month')
    a = {True:.5, False:.3}[annual]
    if labels:
        years = None
    ax.plot(data.index, data.values, 'k+', alpha=a)
    ax.axhline(color='k', lw=.5)
    
    for i in [1, 2]:
        ax.fill_between(data.index, i*sigma, -i*sigma, color='b', alpha=.12)
    labels = label_years(ax, data, sigma, years, labels=labels)
    return labels
    
def max_years(data, years=None):
    if not years:
        years = [1990, 1998, 2016, 2022]
    labels = []
    for yr in years:
        start = yr - 2
        end = yr + 2
        if hasattr(data.index, 'year'):
            start = dt.datetime(start,1,1)
            end = dt.datetime(end,1,1)
        date_range = data.loc[start:end]
        labels.append(date_range.idxmax())
    return labels
        
def label_years(ax, data, sigma, years=None, labels=None):
    """ Label the warmest month in a range centred around the supplied
        years. 
        
        data: Pandas series
        sigma: float
        years: list of ints, years to search for max temperatures
        labels: list of Timestamp values to use with data.loc[], or None
        
        returns: labels
    """
    if not labels:
        labels = max_years(data, years)
    for x in labels:
        y = data.loc[x]
        ys = y/sigma
    if hasattr(x, 'month'):
        t = f"{x:%b %y}\n{ys:.1f}σ"
    else:
        t = f"{x}\n{ys:.1f}σ"
        ax.text(x, y, t, ha='center', 
                va='bottom', size='small')
    return labels
    
# %% Plotting Functions

def plotSlope(annual=False):
    ''' Plot lowess slope showing change because of starting point
    '''
    df = compile_vars(start='1980-01-01')
    yf1 = df.loc['1980-01-01':'2015-01-01']
    yf2 = df.loc['1992-04-01':'2015-01-01']
    yf1.start = 1980
    yf2.start = 1992
    if annual:
        df = df.groupby(df.index.year).mean()
        df.Year = df.index
    period = {True: 'annual', False: 'monthly'}[annual]
    rate = 10
    ax = new_axes('slope', 
                  'Locally Weighted Slope of Global Temperature', 
                  f'{df.spec.name} {period} change from pre-industrial (°C)')
    ax.plot(df.temp, alpha=.3, label='Temperature')
    ax.plot(df.lowess, label='Locally Weighted Slope (LOWESS)')
    m, b = np.polyfit(yf2.Year, yf2.lowess, 1)
    y = m * df.Year + b
    ax.plot(y, label='1990 to 2015 Trend')
    plt.legend()
    
    ax = new_axes('slope_compare',
                  'Slope Estimate Depends on Starting Point',
                  f'{df.spec.name} {period} change from pre-industrial (°C)')
    ax.plot(df.temp, alpha=.3, label='Temperature')
    dashed = (0, (5, 10))
    for yf, c in zip([yf1, yf2], ['C1', 'C2']):
        m, b = np.polyfit(yf.Year, yf.temp, 1)
        x = df.Year.to_numpy()
        y = m * x + b
        label = f'{yf.start} to 2015'
        ax.plot(df.index, y, linestyle=dashed, color=c)
        x = yf.Year.to_numpy()
        y = m * x + b
        ax.plot(yf.index, y, color=c, label=label)
        ax.text(yf.index[0], y[0], f'{m*rate:.3f}°C per decade', 
                va='top', size='large')
    plt.legend()
        
    plt.show()

def plotTempTrend(df=None, adjs=None, annual=False, col='real'):
    """ Plot the monthly temperature trend to 2065
    """
    def get_date(y):
        xpm = xp.max()
        xpi = int((y - intercept) / slope)
        if xpi >= xpm:
            return xpm
        return xpi
    
    def get_slope(xs, ys):
        ''' Return slope and intercept
        
            xs and ys are series
        '''
        x = xs.to_numpy()
        y = ys.to_numpy()
        dx = x[-1] - x[0]
        dy = y[-1] - y[0]
        slope = dy/dx
        intercept = y[0] - slope * x[0]
        return slope, intercept
        
    # Do analysis
    if df is None:
        df = fit_vars(adjs=adjs, annual=annual)
    # Important columns:
        # temp: temperature
        # lowess: local weighted trend calculated from the temperature
        # flat: temperature - lowess
        # vars: natural variation fit to the flat values
        # reduced: detrend - vars
        # real: reduced + lowess
        # trend: linear trend of real
        # detrend: real - trend
    
    start = df.index[0].year
    end = 2065
    xp = np.arange(start, end+1)  # 
    period = 'annual' if annual else 'Monthly'
    alpha = .5 if annual else .3
    
    def plot(y, yt, ytp, win_name='projection', title_app=''):
        """ Plot values against trend and variance
        
            y: values
            yt: trend
            ytp: projected trend
        """
        sigma = (y - yt).std()
    
        dates = {1.5:get_date(1.5), 
                 2.0:get_date(2.0)}
        ymin = 0
        xmin = df.Year.iloc[0]
        slope, intercept = get_slope(df.Year, yt)
        fig = plt.figure(win_name, clear=True)
        ax = fig.add_subplot(111)
        ax.set_ylim(ymin, 2.7)
        ax.plot(df.Year, y, 'k+', alpha=alpha, 
                label=f'{period} Temperature') # data
        ax.plot(xp, ytp, 'b-', lw=1) # trend
        ax.fill_between(xp, ytp+2*sigma, ytp-2*sigma, color='b', alpha=.12)
        ax.fill_between(xp, ytp+sigma, ytp-sigma, color='b', alpha=.12)
        for k in dates.keys():
            year = int(dates[k])
            ax.hlines(k, xmin, dates[k], color='k', lw=0.5, ls=':')
            ax.vlines(dates[k], ymin, k, color='k', lw=0.5, ls=':')
            ax.text(dates[k], k, year, ha='right', va='bottom', weight='bold')

        ax.text(xp[-1], ytp[-1]+sigma*2, '95% Range', va='bottom')
        ax.text(xp[-1], ytp[-1]+sigma, '68% Range', va='center')
        ax.text(xp[-1], ytp[-1], f'σ = {sigma:.3f}°C', va='top')
        text = ("Note: This is a very simplistic projection based only on past trends.\n"+
                "Natural Influences are El Niño, volcanic activity, and solar.")
        ax.text(start, 2.2, text, size='large', ha='left')
        rate = 10
        change = {True:'annual', False:'monthly'}[annual]
        ax.text(get_date(1.75), 1.75, f"{slope*rate:.3f}°C/decade", va='top')
        subtitle = f"{df.spec.name} {change} change from pre-industrial (°C)"
        tls.titles(ax, f"Temperature Projection to {end}",
                   f"{subtitle}{title_app}")
        tls.byline(ax)
        plt.show()
        
        return ax

    tname = f'_{df.index.name}'
         
    #=== Plot trend compared with natural influences ===
    
    slope, intercept = np.polyfit(df.Year, df.temp, 1)
    yt = slope * df.Year + intercept
    ytp = slope * xp + intercept
    ax = plot(df.temp, yt, ytp, win_name='projection_compare'+tname,
              title_app=', Comparing Natural Influences')
    y = (df.vars + df.lowess).values
    ax.plot(df.Year, y, label = 'Natural Influences')
    ax.legend(loc="center left")
    
    #=== Plot Trend with natural influences removed ===
    
    slope, intercept = get_slope(df.Year, df.trend)
    ytp = slope * xp + intercept
    if col == 'real':
        app_txt = ', natural influences removed'
    elif col == 'decreal':
        app_txt = ', natural influences and correlation removed'
    else:
        app_txt = ''
    ax = plot(df[col], df.trend, ytp, win_name='projection_reduced'+tname,
              title_app=app_txt)
    df['smooth'] = df['real'].rolling(12, min_periods=1, center=True).mean()
    ax.plot(df.Year, df.smooth, color='b', lw=1, label='12-month Average')
    ax.legend(loc="center left")

    plt.show()
    return

def plotTempVar(df=None, adjs=None, annual=False):
    """ Plots variability of global temperature with and without
        external variation (ENSO, volcanic sulphates, and solar) removed.
    """
    if df is None:
        df = compile_vars()
    if not hasattr(df, 'detrend'):
        df = fit_vars(df, adjs=adjs, annual=annual)
    slope, intercept = np.polyfit(df.Year, df.temp, 1)
    raw_detrend = df.temp - slope * df.Year - intercept
    labels = max_years(raw_detrend)
    p = {True:'Annual', False:'Monthly'}[annual]
    
    # Plot temperature and Nino Index
    axs = new_fig_rows(f'Deviation-{p}',
                       "Temperature Deviation from Trend",
                       f"{df.spec.name} {p} Global Temperature (ºC)",
                       num=2)
    sigma = raw_detrend.std()
    ax = axs[0]
    plot_one(ax, raw_detrend, sigma, labels=labels)
    # plot natural influences over top
    clr = 'r'
    ax.plot(df.index, df.vars.values, color=clr, lw=1)
    ax.text(df.index[-1], df.vars.iloc[-1], " Natural\n Influences",
            color=clr, size='small', va='center', weight='bold')
    rate = 10  # per decade
    text = f' Trend:\n {slope*rate:.3f}°C/decade\n σ = {sigma:.3f}°C'
    ax.text(df.index[-1], -.01, text, size='small', va='top')
    ylim = ax.get_ylim()
    text = 'Natural influences are El Niño Indexes, Volcanic particles, and Solar'
    ax.text(.99, 0.01, text, ha='right', transform=ax.transAxes,
            color='r', size='small')
    # plot reduced data in next axes
    ax = axs[1]
    clr = 'b'
    nsigma = df.detrend.std()
    df['smooth'] = df.detrend.rolling(12, min_periods=1, center=True).mean()
    plot_one(ax, df.detrend, nsigma, labels=labels)
    ax.plot(df.smooth, color=clr, lw=1)
    ax.set_ylim(ylim)
    
    ax.text(df.index[0], 2.1*nsigma, "Natural Influences Removed",
            color='k', weight='bold')
    dy = df.trend.iloc[-1] - df.trend.iloc[0]
    dx = df.Year.iloc[-1] - df.Year.iloc[0]
    slope = dy/dx
    text = f' Trend:\n {slope*rate:.3f}°C/decade\n σ = {nsigma:.3f}°C'
    ax.text(df.index[-1], -.01, text,
            size='small', va='top')
    ax.text(df.index[-1], nsigma, '12-Month \nAverage', color=clr,
            va='center', size='small', weight='bold')
   
    plt.show()
    return

def plotInfluences(df=None, adjs=None, annual=False):
    """
    Show how the fitted natural influences compare

    Parameters
    ----------
    df : pd.DataFrame, default None
        Dataframe of temperature and natural variances. Gets data from
        `compile_vars()` if not provided.
        
    adjs : pd.DataFrame, optional
        Dataframe of adjustments to make on the influences. Usually the 
        output of `optimize_adjustments()`.
        
    annual : bool, default False
        Returns annual data if true, monthly data otherwise.

    Returns
    -------
    None.

    """
    size = (10, 9)  # size on charts in inches
    if df is None:
        df = compile_vars()
    raw = compile_vars(df.index.name, df.index[0])  # ensures original data used
    if 'reduced' not in df.columns:
        df = fit_vars(df, adjs=adjs, annual=annual)
    e_cols = 'N12 N3 N4 N34'.split()
    cols = 'vol solar enso seasonal sine'.split()
    if 'sine' not in df.columns:
        cols.remove('sine')
    for c in cols[3:]:
        raw[c] = df[c].copy()
    name = 'All Volcanic Solar ENSO Seasonal Sine'.split()
        
    def scale(r, s):
        ''' Return value that scales r to s
        '''
        i = abs(s).idxmax()
        j = abs(r).idxmax()
        c = s[i] / r[j]
        corr = (s * r * c).sum()  # correct sign using correlation
        return c * np.sign(corr)

    # Scale raw data to fitted data
    for c in e_cols:
        raw[c] *= scale(raw[c], df[c])
    raw['enso'] = raw[e_cols].sum(axis=1)
    for c in cols:
        raw[c] *= scale(raw[c], df[c])
    raw['vars'] = raw[cols].sum(axis=1)
    
    # plot major influences
    axs = new_fig_rows(f'Influences_{df.index.name}', 
                       f'Natural Influences on {df.spec.name} Global Temperature', 
                       'Temperature Effect °C', num=len(cols)+1)
    axs[0].figure.set_size_inches(size)
    for i, c in enumerate(['vars']+cols):
        ax = axs[i]
        ax.plot(df[c])
        ax.plot(raw[c], alpha=0.3)
        ax.set_ylim((-.35, .35))
        ax.text(.1, .2, name[i], transform=ax.transAxes, 
                weight='bold')
    l1,  = axs[0].plot([], [], color='C1', alpha=0.3, label='Original Value')
    l2,  = axs[0].plot([], [], color='C0', label='Adjusted Value')
    ax.figure.legend(handles=[l1, l2], loc='lower left', framealpha=0.3,
                  bbox_to_anchor=(.7, .65, .5, .5))
    # Break down the ENSO index
    axs = new_fig_rows(f'ENSO_{df.index.name}',
                       f'ENSO Components Fit to {df.spec.name} Global Temperature',
                       'Temperature Effect °C', num=len(e_cols)+1)
    axs[0].figure.set_size_inches(size)
    names = ['Combined'] + e_cols
    for i, c in enumerate(['enso']+e_cols):
        ax = axs[i]
        ax.plot(df[c])
        ax.plot(raw[c], alpha=.3)
        ax.set_ylim((-.35, .35))
        ax.text(df.index[0], -.1, names[i], weight='bold')
    l1,  = axs[0].plot([], [], color='C1', alpha=0.3, label='Original Value')
    l2,  = axs[0].plot([], [], color='C0', label='Adjusted Value')
    ax.figure.legend(handles=[l1, l2], loc='lower left', framealpha=0.3,
                  bbox_to_anchor=(.7, .69, .5, .5))
    
    plt.show()

def plotHist(df=None, num=3):
    """ Plot histograms of detrended temperature and with natural influences
        removed.
        
        num: number of bins per standard deviation
    """
    def normal(x, mu, sigma):
        y = 1/(sigma * np.sqrt(2 * np.pi)) \
            * np.exp( - (x - mu)**2 / (2 * sigma**2))
        return y
    
    if df is None:
        df = fit_vars()
    cols = ['detrend', 'reduced']
    titles = {cols[0]:'Global Temperature',
              cols[1]:'Natural Influences Removed'}
    fig, axs = plt.subplot_mosaic([cols], num='hist', clear=True, 
                                  sharey=True, layout='tight')
    fig.suptitle('Histograms of Monthly Temperatures',
                 ha='left', x=0.1)
    fig.supxlabel('Deviation from Trend (°C)')
    axs[cols[0]].set_ylabel('Number of Months')
    fig.subplots_adjust(wspace=0.02)
    size = {}
    std = {}
    bcols = {}
    pdf = pd.DataFrame(index=np.arange(-4*num, 4*num+1))
    d = df[cols].copy()
    for c in cols:
        axs[c].set_title(titles[c], loc='left')
        std[c] = d[c].std()
        size[c] = std[c] / num  # bin sizes are relative to std dev
        bcols[c] = 'bins_' + c
        pcols = [c, bcols[c]]
        d[bcols[c]] = d[c] // size[c]  # put each data point in a bin
        pdf[c] = d[pcols].groupby(bcols[c]).count()
        edges = np.arange(pdf.index[0], pdf.index[-1]+2) * size[c]
        axs[c].stairs(pdf[c].values, edges, fill=True)
        x = edges[:-1]+size[c]/2
        axs[c].plot(x, size[c]*len(d)*normal(x, 0, std[c]),
                lw=2)
        if c == cols[0]:
            xlim = axs[c].get_xlim()
        else:
            axs[c].set_xlim(xlim)
    plt.show()
        
def plotWarmingDemo():
    """ Plot ocean warming curve demonstration
    """
    df = pd.DataFrame(index=np.arange(200))
    cols = ['Warming', 'Cooling', 'Vol']
    wn, cn, vn = cols
    rn = ' Result'
    df[wn] = 0.
    # Add a heating step function
    st = 25
    df.loc[df.index > st, wn] = 1.
    # Apply the ocean warming result
    # convolve_impulse converts annual values to a sum of steps
    df[wn+rn] = convolve_step(df[wn])
    # Add a cooling step function
    df[cn] = df[wn]
    df.loc[df.index >= (st+50), cn] = -1.
    df.loc[df.index >= (st+100), cn] = 0.
    df[cn+rn] = convolve_step(df[cn])
    # 
    vol = calc_volcano()
    # normalize
    vol /= -vol.max()
    start = 124
    df[vn] = vol.iloc[start:(start+200)].values
    df[vn+rn] = convolve_step(df[vn])
    # Plot curves
    axs = new_fig_rows('Warming', 
                       'Illustration of Warming in Land and Ocean', '', 3)
    axs[-1].set_xlabel('Years (Months for Eruption)')
    titles = ['A) Warming Curve from Step Rise in Energy Applied',
              'B) Warming and Cooling',
              'C) Cooling from 1991 Mount Pinatubo Eruption (months)']
    lims = [(-0.2, 1.2),
            (-1.2, 1.2),
            (-1.2, 0.2)]
    for ax, col, title, lim in zip(axs, cols, titles, lims):
        ax.plot(df[col], color='C1', ds='steps-post')
        ax.plot(df[col+rn], color='C0')
        ax.set_ylim(lim)
        ax.text(0, lim[1]-.1, title, ha='left', va='bottom', 
                weight='bold')
    adj = 0.02  # text positioning adjustment
    axs[0].text(100, df[wn][100]-adj, 'Temperature Influence', 
                color='C1', va='top')  
    axs[0].text(100, df[wn+rn][100]-adj, 'Resulting Temperature',
                color='C0', va='top')
    
    axs[0].annotate(f'{df[wn+rn][50]*100:.0f}% warming\nafter 20 years',
                    ((st+20), df[wn+rn][st+20]), xytext=(20, -20), 
                    textcoords='offset points', va='top',
                    arrowprops=dict(width=2, headwidth=7, headlength=5))
    plt.show()
        
def plotRate(df=None, col=None, adjs=None, annual=False, stdDevs=2.0,
             name='Slopes', decorrelated=False, verbose=False):
    """ Plot charts determining the global temperature warming rate within
        the dataset to see if there is any statistically relevant acceleration.
        
        verbose: (Bool) If True, prints extra charts for blog
    """
    reduced = True
    if df is None:
        df = compile_vars()
        if adjs is None:
            reduced = False
        else:
            df = fit_vars(df, adjs)
            reduced = True

    if annual:
        df = df.groupby(df.index.year).mean()
        df[yn] = df.index
    else:
        tr.convertYear(df)
    if not col:
        if reduced:
            col = 'real'
            rt = '(natural influences removed)'
        else:
            col = 'temp'
            rt = ''
    elif decorrelated:
        rt = '(natural inflences and autocorrelation removed)'

    # get data as Numpy arrays
    x = df[yn].to_numpy()
    y = df[col].to_numpy()
    lsq = tr.analyzeData(x, y, stdDevs)  # Analyse data
    mx, my = tr.movingAverage(x, y, 12)  # 1-year moving average
    
    # === Plot since start ===
    
    pt = 'Annual' if annual else 'Monthly'  # period text
    pi = 1 if annual else 12  # period index
    
    if verbose:  # plot trend for full period
        ylabel = (f'{df.spec.name} {pt} Change from Pre-Industrial,'+
                  f'°C {rt}')
        ax = new_axes(name='Full Trend',
                      title='Temperature Trend since {df.index[0].year}',
                      ylabel=ylabel)
        ax.plot(mx, my, 'g-', lw=2)        # moving average
        ax.plot(lsq.xline, lsq.yline, 'b-', lw=2)  # trend
        ax.plot(x, lsq.y1, 'b-', lw=1) # lower limit
        ax.plot(x, lsq.y2, 'b-', lw=1) # upper limit
        ax.plot(x, y, 'k+', alpha=(0.3+.3*annual), lw=2)     # data
        # label chart
        error = stdDevs * lsq.sigma
        text = f'Trend: {lsq.slope*10:.3f}±{error*10:.3f} °C/decade'
        ax.text(0.5, 0.25, text, va='top', ma='left', ha='center',
                 transform=ax.transAxes)
    
    # === Plot comparison of 10 vs 20 year trend ===
    
    if verbose:
        ylabel = (f'{df.spec.name} {pt} Change from Pre-Industrial,'+
                  f'°C {rt}')
        axs = new_fig_rows('Compare Trends',
                           title='Comparing Trends with Differing Amounts of Data',
                           ylabel=ylabel,
                           num=2)
        a = df.temp.iloc[-20*pi[annual]:]  # last 20 years
        b = df.temp.iloc[-10*pi[annual]:]  # last 10 years
        for d, ax, txt in zip([a, b], axs, ['20-Year Trend', '10-Year Trend']):
            x = d[yn].to_numpy()
            y = d[col].to_numpy()
            lsq = tr.analyzeData(x, y, stdDevs)  # Analyse data
            ax.plot(x, y, 'k+', alpha=(0.3+.3*annual), lw=1)     # data
            if not annual:
                mx, my = tr.movingAverage(x, y, 1*12)  # 1-year moving average
                ax.plot(mx, my, 'g-', lw=2)        # moving average
            ax.plot(lsq.xline, lsq.yline, 'b-', lw=2)  # trend
            ax.plot(x, lsq.y1, 'b-', lw=1) # lower limit
            ax.plot(x, lsq.y2, 'b-', lw=1) # upper limit
            # label chart
            error = stdDevs * lsq.sigma
            text = f'{txt}: {lsq.slope*10:.3f}±{error*10:.3f} °C/decade'
            ax.text(0.5, 0.15, text, va='top', ma='left', ha='center',
                     transform=ax.transAxes)

    # === Plot trends from moving breakpoint ===
    
    # set up data store for calculations
    columns = ['before', 'bhi', 'blo', 'after', 'ahi', 'alo',
               'bnu', 'bsy', 'bsxy', 
               'anu', 'asy', 'asxy']
    sides = columns[0:6:3]
    highs = columns[1:6:3]
    lows = columns[2:6:3]
    nus = columns[6::3]
    sys = columns[7::3]
    sxys = columns[8::3]
    lim = 5 * (12 - 11*annual)  # minimum number of months/years for slope
    stats = pd.DataFrame(index=df.index[lim:-lim], 
                         columns=columns, dtype=np.float64)
    stats[yn] = df[yn]
    
    # calculate trend lines before and after d
    cols = [col, yn]
    unit = 10.  # °C per decade using data in fractions of a year
    for d in stats.index:
        t1 = df.loc[df.index < d, cols]
        t2 = df.loc[df.index >= d, cols]
        for t, side, hi, lo, nu, sy, sxy in zip([t1, t2], sides, highs, lows,
                                   nus, sys, sxys):
            x = t[yn].to_numpy()
            y = t[col].to_numpy()
            lsq = tr.analyzeData(x, y, stdDevs)  # Analyse data
            dev = stdDevs * lsq.sigma * unit
            stats.loc[d, side] = lsq.slope * unit
            stats.loc[d, hi] = lsq.slope * unit + dev
            stats.loc[d, lo] = lsq.slope * unit - dev
            stats.loc[d, nu] = lsq.nu
            stats.loc[d, sy] = lsq.sy
            stats.loc[d, sxy] = lsq.sxy
            
    # plot the before and after slopes
    pt2 = {True:'Year', False:'Month'}[annual]
    ax = new_axes(name=name,
                  title=f'Comparing Trends Before and After Each {pt2}',
                  ylabel=f'{df.spec.name} Trend in °C per decade {rt}')
    ax.plot(stats.before, '-', color='C0', lw=3, label='Trend Before Date')
    ax.fill_between(stats.index, stats.blo, stats.bhi, color='C0', alpha=0.25)
    ax.plot(stats.after, '-', color='C1', lw=3, label='Trend After Date')
    ax.fill_between(stats.index, stats.alo, stats.ahi, color='C1', alpha=0.25)
    ax.set_ylim(-.2, .4)
    ax.plot([], [], 'k-', lw=10, alpha=0.15, label='95% Confidence Ranges')
    ax.legend(loc='upper center')
    
    # === plot histograms for lowest overlap ===
    
    stats['overlap'] = stats.bhi - stats.alo
    imin = stats.overlap.loc[stats.Year>2000].idxmin() 
    print(f'\nMost likely break point: {imin}')
    # print(stats.loc[imin])  # data for imin

    plt.show()
    return
        
def plotBreak(point, data, continuous=True,
              annual=False, rt='', stdDevs=2., ct=False):
    ''' Plot the slopes before and after a supplied break point. The point
        must be in integer or floating point years (fractional).
        
        continuous: if True, use a continuous fit
        ct: if True, run Chow Test for no break. Not useful for correlated
            data.
    '''
    df = pd.DataFrame(data.values, data.index, columns=[dn])
    if annual and hasattr(df.index, 'year'):
        df = df.groupby(df.index.year).mean()
        df[yn] = df.index
    else:
        tr.convertYear(df)
    
    rt1 = '_Reduced' if (rt != '') else ''
    pt = 'Annual' if annual else 'Monthly'  # period text

    ax = new_axes(name=f'Break_{pt}{rt1}_{point}',
                  title=f'Comparing {pt} Trends Before and After {point}',
                  ylabel=f'Temperature Trend in °C per decade {rt}')
    t0 = df.loc[df[yn] < point]
    t1 = df.loc[df[yn] >= point]
    t2 = df 
    labels = ['Before', 'After', 'All']
    colors = ['C0', 'C1', 'w']
    alphas = [1, 1, 0]
    unit = 10  # °C/decade
    md = [0, 1, 2]  # metadata for before and after
    
    # === Plot rates for before and after ===
    
    for t, idx, clr, label in zip([t0, t1, t2], [0,1,2], colors, labels):
        x = t[yn].to_numpy()
        y = t[dn].to_numpy()
        if continuous and idx==1:  # Use a continuous fit
            knot = (md[0].xline[-1], md[0].yline[-1])  # last point in fit
            lsq = tr.analyzeData(x, y, knot=knot)
        else:
            lsq = tr.analyzeData(x, y, stdDevs)  # Analyse data
        md[idx] = lsq
        if idx==2: 
            continue
        x = [t.Year.iloc[0], t.Year.iloc[-1]]
        y = [lsq.slope * unit, lsq.slope * unit]
        ax.plot(x, y, '-', color=clr, lw=2, label=label)
        err = lsq.sigma * stdDevs * unit
        ax.fill_between(x, y-err, y+err, color=clr, alpha=0.15)
    tr.analyzeRate(df, dn, window=30, stdDevs=stdDevs)
    x = df[yn].to_numpy()
    y = df.Rate.to_numpy() * unit
    ax.plot(x, y, 'k-', lw=1, label='Rate with 20-year window')
    ax.fill_between(x, df.R1*unit, df.R2*unit, color='k', alpha=.05)
    ax.set_ylim(0, .6)
    ax.plot([], [], 'k-', lw=10, alpha=0.15, label='95% Confidence Ranges')
    ax.legend(loc='upper left')
    
    # === Plot temperature with these trends ===
    
    ax = new_axes(name=f'Slopes_{pt}{rt1}_{point}',
                  title=f'Comparing {pt} Trends Before and After {point:.0f}',
                  ylabel=f'{pt} Temperature change from pre-industrial (°C) {rt}')
    md[0].xline[1] = df.Year.iloc[-1]
    preps = [f'before {point:.0f}', f'after {point:.0f}', 'for all data']
    for lsq, clr, a, prep in zip(md, colors, alphas, preps):
        ax.plot(lsq.x, lsq.y1, '-', color=clr, lw=0.5)
        ax.plot(lsq.x, lsq.y2, '-', color=clr, lw=0.5)
        yline = lsq.slope * lsq.xline + lsq.intercept
        label = f'Trend {prep}: {lsq.slope*unit:.3f}°C per decade'
        ax.plot(lsq.xline, yline, '-', color=clr, alpha=a, lw=2, label=label)
    ax.plot(df[yn].to_numpy(), df[dn].to_numpy(), 'k+', alpha=0.4)
    ax.legend(loc='upper left')
    lsq = tr.analyzeData(df[yn].to_numpy(),
                         df[dn].to_numpy(),
                         stdDevs)
    if ct:
        ch = chow(lsq.res, md[0].res, md[1].res, 2)
        hyp = {True:'Pass above', False:'Fail below'}[(ch.p > 0.05)]
        text = ('Chow test for no break: \n' +
                f'Test: {ch.Chow:0.1f} (DF₁: {ch.DoF1:.0f}, DF₂: {ch.DoF2:.0f})\n' +
                f'p-value: {ch.p:0.3f} ({hyp} 0.05 threshold)')
        ax.text(.2, .7, text, transform=ax.transAxes,
                ha='left')

    plt.show()
    
def plotARDemo(df=None, adjs=None, N=24, start='1990-01-01'):
    '''
    Demonstrate autocorrelation in temperature data
    
    Parameters
    ----------
    
    df : pandas.DataFrame
        (Optional) Dataframe of fitted and detrended temperature data. If 
        not provided, will create one.
        
    adjs : pandas.DataFrame
        (Optional) Datafram of adjustments to make to natural influences
        before fitting to temperature. If not provided, will determine best
        values, but the process is takes a few minutes.
    
    N : int (Optional, default 24)
        The number of lags (in months) to use in creating the autoregression 
        model.
        
    start : str (Optional, default "1990-01-01")
        The start date if compiling the fitted data. Not used if the data 
        is provided in `df`.
    '''
    # Height in inches of short or tall figures (default: 6.75)
    fig_short = 5.5
    if df is not None:
        df = compile_vars(start=start)
        df = fit_vars(df, adjs=adjs)
    # obs is detrended data with natural influences removed
    obs = pd.Series(index=pd.date_range(df.index[0], df.index[-1], freq='MS'))
    obs.loc[:] = df.detrend[obs.index].to_numpy()
    obs -= obs.mean()
    
    # Plot detrended temperature
    ax = new_axes('stationary', 
                  'Global Temperature with Trend and Natural Influences Removed', 
                  f'{df.spec.name} monthly change from trend (°C)')
    ax.figure.set_figheight(fig_short)
    smooth = tr.lowess(obs, pts=15)
    ax.plot(obs, '+k', alpha=.3)
    ax.plot(smooth, color='C1', label='15-Month Locally Weighted Smoothing (LOWESS)')
    ax.text(df.index[-1], -.15, f'Trend: {df.fit["slope"]*10:0.3f}°C/decade', 
            ha='right', va='top')
    ax.set_ylim((-.34, .34))
    ax.legend()
    
    # Plot autocoorelation lags to show that data is still autocorrelated.
    ax = new_axes('autocorrelation', 
                  'Autocorrelation of Global Monthly Temperature', 
                  '')
    ax.figure.set_figheight(fig_short)
    ax.set_xlabel('Lag')
    plot_acf(obs, ax, lags=N, title='', adjusted=True, alpha=.15)
    ax.set_ylim([-.25, 1.19])
    
    # Plot partial autocorrelation to estimate number of AR parameters needed
    ax = new_axes('partial', 
                  'Partial Autocorrelation of Global Temperature', 
                  '')
    ax.figure.set_figheight(fig_short)
    ax.set_xlabel('Lag')
    plot_pacf(obs, ax, N, title='', method='yw', alpha=.15)  # adjusted Yule-Walker
    ax.set_ylim([-.25, 1.19])
    
    # Get model parameters
    res = get_ARIMA(obs, N)
    params = res.params.drop(index=['ma.L1'])
    sigma = params.sigma2 ** 0.5  # get standard deviation of input noise
    lags = var2lag(params)  # make variable names into lags
    print(f'\nLags\n{lags}\n')
    
    # Create model
    arima = sm.tsa.arima.ARIMA
    model = arima(obs, order=(lags, 0, 0), dates=obs.index, trend='n')
    sim = model.simulate(params, len(obs))
    axs = new_fig_rows('compare', 
        'Comparison of Temperature and Model', 
        '', 2, labels=['Temperature (trend and natural influences removed)',
                       'Autocorrelation Model'])
    axs[0].plot(obs)
    axs[1].plot(sim)
    
    # Test non-stationary input data
    params.drop('sigma2', inplace=True)
    test = df.loc[df.Year>=2005, ['Year']].copy()
    test['line'] = 0.
    slope = 0.015
    late =test.loc[test.Year>=2010].index
    test.loc[late, 'line'] = slope * (test.Year[late] - 2010)
    test['data'] = test.line + np.random.normal(0, sigma, len(test))
    test['AR'] = AR_process(test.data, params)
    test['decorr'] = decorrelate(test.AR, params)
    axs = new_fig_rows('compare_test', 
        'Comparison of Test Noise and Model Output', 
        '', 3, labels=['Noise', 'Correlated', 'Decorrelated'])
    axs[0].plot(test.data)
    axs[1].plot(test.AR)
    axs[2].plot(test.decorr)
    lims = pd.DataFrame(index=np.arange(3), columns=['lo', 'hi'])
    for i in range(3):
        axs[i].plot(test.line)
        lims.loc[i, :] = axs[i].get_ylim()
    for i in range(3):
        axs[i].set_ylim((lims.lo.min(), lims.hi.max()))
        
    
    df['decorr'] = decorrelate(df.detrend, params)
    axs = new_fig_rows('compare_temp', 
        'Comparison of Temperature and Decorrelated Data', 
        f"{df.spec.name} monthly (°C), with trend and natural influences removed", 
        2, labels=['', 'Decorrelated'])
    for ax, data in zip(axs, [df.detrend, df.decorr]):
        smooth = tr.lowess(data, pts=15)
        ax.plot(data, '+k', alpha=.3, label='Monthly')
        ax.plot(smooth, color='C1', label='LOWESS')
        ax.set_ylim((-.34, .34))
    axs[1].legend(loc='upper center')
    
    ax = new_axes('acf_decorrelated', 
                  'Autocorrelation of Decorellated Temperature', 
                  '')
    ax.figure.set_figheight(fig_short)
    ax.set_xlabel('Lag')
    plot_acf(df.decorr, ax, lags=N, title='', adjusted=True)
    ax.set_ylim([-.25, 1.19])
    
    df['decreal'] = df.decorr + df.trend
    
    plotRate(df, col='decreal', decorrelated=True)
    plotBreak(2015, df.decreal, 
              rt='(natural influences and autocorrelation removed)')  
    plotTempTrend(df, adjs, col='decreal')
    
    plt.show()
    
# %% Notes
"""
# Startup steps to recreate working data
# Be sure to run the necessary files in the iPython window:
    datastore.py
    qplot.py
    projection.py
tmp = 'giss'
ds.update_modern(tmp)
ds.update_modern('enso')
df = compile_vars(source=tmp)
adj = optimize_adjustments(df)  # so you only have to do this once
plotInfluences(df, adj)
plotTempVar(df, adj)  # check data

"""
    

